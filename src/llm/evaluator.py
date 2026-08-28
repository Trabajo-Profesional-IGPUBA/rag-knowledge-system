from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient, LLMConfig
from src.llm.prompt_builder import PromptBuilder
from src.llm.rag_pipeline import RAGConfig, RAGPipeline
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)


EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "id": "q1",
        "query": "¿Cuál es la presión de fondo del pozo PM-104?",
        "expected_keywords": ["presión", "psi", "fondo", "PM-104"],
        "category": "datos_técnicos",
    },
    {
        "id": "q2",
        "query": "¿Qué problemas se reportaron en los partes diarios del último workover?",
        "expected_keywords": ["workover", "problema", "reporte", "parte"],
        "category": "operaciones",
    },
    {
        "id": "q3",
        "query": "¿Cuál es la profundidad total del pozo y su formación productiva?",
        "expected_keywords": ["profundidad", "metros", "formación", "productiva"],
        "category": "geología",
    },
    {
        "id": "q4",
        "query": "Describí el procedimiento de estimulación utilizado.",
        "expected_keywords": ["estimulación", "procedimiento", "tratamiento"],
        "category": "operaciones",
    },
    {
        "id": "q5",
        "query": "¿Qué producción de gas se registró en el último parte diario?",
        "expected_keywords": ["gas", "producción", "m3", "parte diario"],
        "category": "producción",
    },
]


@dataclass
class ModelEvalResult:
    model: str
    query_id: str
    query: str
    response: str
    elapsed_sec: float
    response_length: int
    keyword_hits: int
    keyword_total: int
    keyword_score: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class EvaluationReport:
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    models_evaluated: list[str] = field(default_factory=list)
    results: list[ModelEvalResult] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    selected_model: str = ""
    selection_rationale: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Reporte de evaluación guardado en %s", path)

    def print_summary(self) -> None:
        print("\n" + "═" * 60)
        print("EVALUACIÓN COMPARATIVA DE MODELOS LLM")
        print("═" * 60)
        for model, stats in self.summary.items():
            print(f"\n  Modelo: {model}")
            print(f"    Latencia promedio : {stats['avg_elapsed_sec']:.2f}s")
            print(f"    Score keywords    : {stats['avg_keyword_score']:.0%}")
            print(f"    Resp. promedio    : {stats['avg_response_length']:.0f} chars")
            print(f"    Errores           : {stats['error_count']:.0f}")
        print(f"\n  → Modelo seleccionado: {self.selected_model}")
        print(f"  → Justificación: {self.selection_rationale}")
        print("═" * 60)


def _score_keywords(response: str, keywords: list[str]) -> tuple[int, float]:
    """Cuenta keywords esperadas presentes en la respuesta (case-insensitive)."""
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    score = hits / len(keywords) if keywords else 0.0
    return hits, score


def evaluate_models(
    retriever: Retriever,
    models: list[str],
    queries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
) -> EvaluationReport:
    queries = queries or EVAL_QUERIES
    report = EvaluationReport(models_evaluated=models)
    prompt_builder = PromptBuilder()

    for model_name in models:
        log.info("Evaluando modelo: %s", model_name)
        print(f"\n[Evaluando {model_name}...]")

        config = LLMConfig(model=model_name, temperature=0.1)
        client = LLMClient(config)

        if not client.is_available():
            log.error("Ollama no disponible para modelo %s", model_name)
            continue

        # Verificar que el modelo esté descargado
        available = client.list_models()
        if model_name not in available:
            log.info("Modelo %s no encontrado, descargando...", model_name)
            client.pull_model(model_name)

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=client,
            prompt_builder=prompt_builder,
            config=RAGConfig(top_k=5, min_score=0.2),
        )

        for q in queries:
            print(f"  → {q['id']}: {q['query'][:50]}...")

            try:
                t0 = time.perf_counter()
                rag_resp = pipeline.query(q["query"])
                elapsed = time.perf_counter() - t0

                hits, score = _score_keywords(
                    rag_resp.answer,
                    q.get("expected_keywords", []),
                )

                result = ModelEvalResult(
                    model=model_name,
                    query_id=q["id"],
                    query=q["query"],
                    response=rag_resp.answer,
                    elapsed_sec=round(elapsed, 2),
                    response_length=len(rag_resp.answer),
                    keyword_hits=hits,
                    keyword_total=len(q.get("expected_keywords", [])),
                    keyword_score=round(score, 4),
                )

            except Exception as e: # noqa: BLE001 — se captura todo para no interrumpir la evaluación de otros modelos/queries
                result = ModelEvalResult(
                    model=model_name,
                    query_id=q["id"],
                    query=q["query"],
                    response="",
                    elapsed_sec=0.0,
                    response_length=0,
                    keyword_hits=0,
                    keyword_total=len(q.get("expected_keywords", [])),
                    keyword_score=0.0,
                    error=str(e),
                )
                log.error("Error evaluando %s en %s: %s", model_name, q["id"], e)

            report.results.append(result)

    for model_name in models:
        model_results = [r for r in report.results if r.model == model_name]
        if not model_results:
            continue

        ok_results = [r for r in model_results if r.ok]
        report.summary[model_name] = {
            "avg_elapsed_sec": (
                round(sum(r.elapsed_sec for r in ok_results) / len(ok_results), 2)
                if ok_results
                else 0.0
            ),
            "avg_keyword_score": (
                round(sum(r.keyword_score for r in ok_results) / len(ok_results), 4)
                if ok_results
                else 0.0
            ),
            "avg_response_length": (
                round(sum(r.response_length for r in ok_results) / len(ok_results), 1)
                if ok_results
                else 0.0
            ),
            "error_count": len([r for r in model_results if not r.ok]),
            "total_queries": len(model_results),
        }

    if report.summary:
        best = max(
            report.summary.items(),
            key=lambda x: (x[1]["avg_keyword_score"], -x[1]["avg_elapsed_sec"]),
        )
        report.selected_model = best[0]
        stats = best[1]
        report.selection_rationale = (
            f"Mayor score de relevancia ({stats['avg_keyword_score']:.0%}) "
            f"con latencia de {stats['avg_elapsed_sec']:.1f}s promedio. "
            f"Mejor balance entre calidad de respuesta y rendimiento en CPU."
        )

    if output_path:
        report.save(output_path)

    report.print_summary()
    return report
