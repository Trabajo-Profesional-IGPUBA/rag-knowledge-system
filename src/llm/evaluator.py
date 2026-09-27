"""Evaluación comparativa de modelos LLM sobre el pipeline RAG (latencia, cobertura de keywords)."""

import json
import logging
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient, LLMConfig
from src.llm.prompt_builder import PromptBuilder
from src.llm.rag_pipeline import RAGConfig, RAGPipeline
from src.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

NO_INFO_PATTERNS = [
    "no encontré información",
    "no tengo información",
    "no se proporcionan detalles",
    "no encontré datos",
    "no dispongo de información",
]

EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "id": "q1",
        "query": "¿Tuvimos problemas de pérdida de circulación en la formación Quintuco?",
        "expected_keywords": ["pérdida de circulación", "Quintuco", "LCM", "PM-104"],
        "category": "perforación",
    },
    {
        "id": "q2",
        "query": "¿Qué pasó con la sarta de varillas en el pozo LL-205?",
        "expected_keywords": ["varillas", "pesca", "overshot", "LL-205", "fatiga"],
        "category": "workover",
    },
    {
        "id": "q3",
        "query": "¿Qué causó el screen-out durante la fractura hidráulica en CH-45?",
        "expected_keywords": [
            "screen-out",
            "arenamiento",
            "presión",
            "CH-45",
            "estimulación",
        ],
        "category": "estimulación",
    },
    {
        "id": "q4",
        "query": "¿Qué mecanismo de corrosión afectó al tubing del pozo YPF-X2?",
        "expected_keywords": [
            "corrosión",
            "CO2",
            "bacterias sulfato-reductoras",
            "tubing",
            "Water Cut",
        ],
        "category": "integridad",
    },
    {
        "id": "q5",
        "query": "¿Hubo canalización preferencial entre el inyector PI-08 y algún pozo productor?",
        "expected_keywords": ["trazador", "canalización", "PI-08", "PM-102", "barrido"],
        "category": "reservorio_inyección",
    },
    {
        "id": "q6",
        "query": "¿Tuvimos problemas con el cable de la BES?",
        "expected_keywords": [
            "VSD",
            "aislamiento",
            "caja de venteo",
            "BES",
            "cable de potencia",
        ],
        "category": "BES",
    },
    {
        "id": "q7",
        "query": "¿A qué profundidad falló la tubería en el pozo LP-15?",
        "expected_keywords": ["tubing", "junta", "metros", "erosión", "LP-15"],
        "category": "workover",
    },
    {
        "id": "q8",
        "query": "¿Qué problemas tuvimos con las BES en este yacimiento por baja tasa de flujo?",
        "expected_keywords": [
            "downthrust",
            "Run Life",
            "sobrecalentamiento",
            "declinación",
            "BES",
        ],
        "category": "BES",
    },
    {
        "id": "q9",
        "query": "¿Por qué el pozo PM-104 está produciendo más gas si no se tocó el estrangulador?",
        "expected_keywords": [
            "presión de burbuja",
            "gas disuelto",
            "PM-104",
            "GOR",
            "PVT",
        ],
        "category": "reservorio",
    },
    {
        "id": "q10",
        "query": "¿Por qué el pozo PM-108 tiene baja productividad crónica?",
        "expected_keywords": [
            "facies",
            "arcillosa",
            "canales fluviales",
            "PM-108",
            "conectividad",
        ],
        "category": "geología",
    },
    {
        "id": "q11",
        "query": "¿Qué falla mecánica ocurrió en el packer del pozo inyector PI-44?",
        "expected_keywords": [
            "packer",
            "incrustaciones",
            "sulfato de bario",
            "PI-44",
            "tracción",
        ],
        "category": "workover",
    },
]


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


def _score_keywords(response: str, keywords: list[str]) -> tuple[int, float]:
    """Cuenta keywords esperadas presentes en la respuesta (case/acentos-insensitive)."""
    response_norm = _normalize(response)
    hits = sum(1 for kw in keywords if _normalize(kw) in response_norm)
    score = hits / len(keywords) if keywords else 0.0
    return hits, score


def _is_no_info_response(response: str) -> bool:
    """Detecta si la respuesta es una abstención ('no encontré información...')."""
    response_norm = _normalize(response)
    return any(_normalize(p) in response_norm for p in NO_INFO_PATTERNS)


def evaluate_models(
    retriever: Retriever,
    models: list[str],
    queries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
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
                    is_no_info_response=_is_no_info_response(rag_resp.answer),
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
            f"Tasa de abstención: {stats['no_info_rate']:.0%}."
        )

    if output_path:
        report.save(output_path)

    report.print_summary()
    return report
