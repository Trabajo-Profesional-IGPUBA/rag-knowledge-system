"""Evaluación comparativa de modelos LLM sobre el pipeline RAG (latencia, cobertura de keywords)."""

import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.embedder import Embedder
from src.llm.client import LLMClient, LLMConfig
from src.llm.eval_queries import EVAL_QUERIES, NO_INFO_PATTERNS
from src.llm.prompt_builder import PromptBuilder
from src.llm.rag_pipeline import RAGConfig, RAGPipeline
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

QUALITY_WEIGHTS = {"keywords": 0.25}


@dataclass
class ModelEvalResult:
    """Resultado de evaluar un modelo sobre una consulta puntual."""

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
    is_no_info_response: bool = False
    semantic_similarity: float | None = None
    quality: float = 0.0

    @property
    def ok(self) -> bool:
        """True si la consulta se ejecutó sin error."""
        return self.error is None


@dataclass
class EvaluationReport:
    """Reporte consolidado de la evaluación: resultados por query, resumen por modelo y selección final."""

    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    models_evaluated: list[str] = field(default_factory=list)
    results: list[ModelEvalResult] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    selected_model: str = ""
    selection_rationale: str = ""

    def to_dict(self) -> dict:
        """Convierte el reporte a diccionario serializable."""
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        """Guarda el reporte como JSON en el path indicado, creando carpetas si hace falta."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Reporte de evaluación guardado en %s", path)

    def print_summary(self) -> None:
        """Imprime en consola un resumen legible del reporte por modelo."""
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


def _normalize(text: str) -> str:
    """Normaliza texto: minúsculas y sin acentos, para matching más tolerante."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text


def _as_groups(keywords: list) -> list[list[str]]:
    """Convierte ['a', ['b', 'c']] en [['a'], ['b', 'c']] (grupos de sinónimos)."""
    return [[k] if isinstance(k, str) else list(k) for k in keywords]


def _score_keywords(
    text: str, keywords: list, query: str = ""
) -> tuple[int, int, float]:
    q_norm = _normalize(query)
    t_norm = _normalize(text)
    groups = [
        g for g in _as_groups(keywords) if not any(_normalize(a) in q_norm for a in g)
    ]
    hits = sum(1 for g in groups if any(_normalize(a) in t_norm for a in g))
    total = len(groups)
    return hits, total, (hits / total if total else 0.0)


def _nums(text: str) -> set[str]:
    """Números sin separadores ('2.450' == '2450'); ignora los de 1 dígito."""
    found = re.findall(r"\d+(?:[.,]\d+)*", text)
    cleaned = {n.replace(".", "").replace(",", "") for n in found}
    return {n for n in cleaned if len(n) >= 2}


def _score_numbers(text: str, reference: str | None, query: str = "") -> float | None:
    """Fracción de los números de la referencia que aparecen en `text`."""
    if not reference:
        return None
    ref = _nums(reference) - _nums(query)
    if not ref:
        return None
    return len(ref & _nums(text)) / len(ref)


def _run_quality(r: ModelEvalResult) -> float:
    """Calidad 0-1 de una corrida. Error = 0."""
    if not r.ok:
        return 0.0

    components: dict[str, float] = {}
    if r.keyword_total > 0:
        components["keywords"] = r.keyword_score

    if not components:
        return 0.0
    weight_sum = sum(QUALITY_WEIGHTS[k] for k in components)
    return sum(QUALITY_WEIGHTS[k] * v for k, v in components.items()) / weight_sum


def _is_no_info_response(response: str) -> bool:
    """Detecta si la respuesta es una abstención ('no encontré información...')."""
    response_norm = _normalize(response)
    return any(_normalize(p) in response_norm for p in NO_INFO_PATTERNS)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores."""
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _semantic_similarity(
    response: str, reference: str | None, embedder: Embedder | None
) -> float | None:
    """Similitud semántica (coseno) entre la respuesta generada y la referencia."""
    if not reference or embedder is None:
        return None
    resp_emb = embedder.embed(response)
    ref_emb = embedder.embed(reference)
    return round(_cosine_similarity(resp_emb, ref_emb), 4)


def _select_best_model(summary: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Selecciona el mejor modelo por calidad combinada (keywords + similitud
    semántica), desempatando por menor latencia. Devuelve (modelo, justificación)."""
    if not summary:
        return "", ""

    def _quality(stats: dict[str, float]) -> float:
        return (stats["avg_keyword_score"] + stats["avg_semantic_similarity"]) / 2

    best_model, stats = max(
        summary.items(),
        key=lambda x: (_quality(x[1]), -x[1]["avg_elapsed_sec"]),
    )
    rationale = (
        f"Mayor calidad combinada (keywords {stats['avg_keyword_score']:.0%} "
        f"+ similitud semántica {stats['avg_semantic_similarity']:.0%}) "
        f"con latencia de {stats['avg_elapsed_sec']:.1f}s promedio. "
        f"Tasa de abstención: {stats['no_info_rate']:.0%}."
    )
    return best_model, rationale


def evaluate_models(
    retriever: Retriever,
    models: list[str],
    queries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
    embedder: Embedder | None = None,
) -> EvaluationReport:
    """Evalúa cada modelo contra el set de queries, arma el resumen y selecciona el mejor."""
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

                hits, total, score = _score_keywords(
                    rag_resp.answer,
                    q.get("expected_keywords", []),
                )

                sem_sim = _semantic_similarity(
                    rag_resp.answer,
                    q.get("reference_answer"),
                    embedder,
                )

                result = ModelEvalResult(
                    model=model_name,
                    query_id=q["id"],
                    query=q["query"],
                    response=rag_resp.answer,
                    elapsed_sec=round(elapsed, 2),
                    response_length=len(rag_resp.answer),
                    keyword_hits=hits,
                    keyword_total=total,
                    keyword_score=round(score, 4),
                    is_no_info_response=_is_no_info_response(rag_resp.answer),
                    semantic_similarity=sem_sim,
                )

            except Exception as e:
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
                    is_no_info_response=False,
                )
                log.error("Error evaluando %s en %s: %s", model_name, q["id"], e)

            report.results.append(result)

    for model_name in models:
        model_results = [r for r in report.results if r.model == model_name]
        if not model_results:
            continue

        ok_results = [r for r in model_results if r.ok]
        sim_values = [
            r.semantic_similarity
            for r in ok_results
            if r.semantic_similarity is not None
        ]
        no_info_count = sum(1 for r in ok_results if r.is_no_info_response)

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
            "avg_semantic_similarity": (
                round(sum(sim_values) / len(sim_values), 4) if sim_values else 0.0
            ),
            "avg_response_length": (
                round(sum(r.response_length for r in ok_results) / len(ok_results), 1)
                if ok_results
                else 0.0
            ),
            "no_info_rate": (
                round(no_info_count / len(ok_results), 4) if ok_results else 0.0
            ),
            "error_count": len([r for r in model_results if not r.ok]),
            "total_queries": len(model_results),
        }

    if report.summary:
        report.selected_model, report.selection_rationale = _select_best_model(
            report.summary
        )

    if output_path:
        report.save(output_path)

    report.print_summary()
    return report
