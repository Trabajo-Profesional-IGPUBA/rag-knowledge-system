"""Dataclasses de la evaluación."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


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
    run: int = 1
    error: str | None = None
    is_no_info_response: bool = False
    semantic_similarity: float | None = None
    quality: float = 0.0
    number_score: float | None = None
    judge_correctness: float | None = None
    judge_completeness: float | None = None  # 1-5
    judge_faithfulness: float | None = None  # 1-5
    should_abstain: bool = False
    abstained: bool = False
    context_recall: float | None = None
    context_ok: bool | None = None
    context: str = ""

    @property
    def ok(self) -> bool:
        """True si la consulta se ejecutó sin error."""
        return self.error is None


@dataclass
class EvaluationReport:
    """Reporte consolidado de la evaluación: resultados por query, resumen por modelo y selección final."""

    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
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
            halluc = (
                f"{stats['hallucination_rate']:.0%}"
                if stats["total_negative"]
                else "n/d"
            )
            print(f"\n  Modelo: {model}")
            print(f"    Calidad global (errores=0) : {stats['quality']:.0%}")
            print(f"    Calidad (contexto con resp.): {stats['quality_ctx_ok']:.0%}")
            print(
                f"    Keywords / Números         : {stats['avg_keyword_score']:.0%} / {stats['avg_number_score']:.0%}"
            )
            print(
                f"    Judge correct. / fidelidad : {stats['avg_judge_correctness']:.1f} / {stats['avg_judge_faithfulness']:.1f} (de 5)"
            )
            print(f"    Resp. promedio    : {stats['avg_response_length']:.0f} chars")
            print(
                f"    Latencia p50 / p95         : {stats['p50_elapsed_sec']:.2f}s / {stats['p95_elapsed_sec']:.2f}s"
            )
            print(
                f"    Abstención incorrecta      : {stats['false_abstention_rate']:.0%}"
            )
            print(f"    Alucinación (sin respuesta): {halluc}")
            print(f"    Recall del retriever       : {stats['retrieval_recall']:.0%}")
            print(
                f"    Errores                    : {stats['error_count']:.0f} de {stats['total_runs']:.0f}"
            )
        print(f"\n  → Modelo seleccionado: {self.selected_model}")
        print(f"  → Justificación: {self.selection_rationale}")
        print("═" * 60)
