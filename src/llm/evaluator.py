"""Evaluación comparativa de modelos LLM sobre el pipeline RAG (latencia, cobertura de keywords)."""

import json
import logging
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.embedder import Embedder
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
        "reference_answer": (
            "Sí, en 2012 el pozo PM-104 tuvo una pérdida de circulación severa "
            "de aproximadamente 15 m³/h en la formación Quintuco, a los 2.450 "
            "metros mdf. Se solucionó bombeando píldoras de LCM (cascarilla de "
            "nuez y carbonato de calcio), y se recomendó no superar un peso de "
            "lodo de 1.15 g/cm³ en pozos colindantes."
        ),
        "category": "perforación",
    },
    {
        "id": "q2",
        "query": "¿Qué pasó con la sarta de varillas en el pozo LL-205?",
        "expected_keywords": ["varillas", "pesca", "overshot", "LL-205", "fatiga"],
        "reference_answer": (
            "En el pozo LL-205 (2019) se produjo un desprendimiento de varillas "
            "a los 1.820 metros por rotura de fatiga en el cuello de una varilla "
            'de 7/8" grado D. Se recuperó con una herramienta de pesca tipo '
            'overshot de 2 3/8", tras limpiar parafina con gasoil caliente. Se '
            "recomendó instalar centralizadores de alta temperatura en el tramo "
            "desviado."
        ),
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
        "reference_answer": (
            "En el pozo CH-45 (2021), durante la etapa 4 de estimulación "
            "hidráulica en la formación Agrio, un incremento repentino de "
            "presión (de 6.200 a 8.500 psi) al ingresar arena mesh 20/40 "
            "provocó un screen-out (arenamiento) prematuro, originado por una "
            "falla mecánica en una bomba de la unidad fracturadora."
        ),
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
        "reference_answer": (
            "El tubing del pozo YPF-X2 (2017) sufrió corrosión por flujo "
            "bifásico con alta concentración de CO2 disuelto y presencia de "
            "bacterias sulfato-reductoras (BSR), agravada por un Water Cut "
            "superior al 85%, con pérdida de sección de hasta el 65% del "
            "espesor de pared entre 1.100 y 1.250 metros."
        ),
        "category": "integridad",
    },
    {
        "id": "q5",
        "query": "¿Hubo canalización preferencial entre el inyector PI-08 y algún pozo productor?",
        "expected_keywords": ["trazador", "canalización", "PI-08", "PM-102", "barrido"],
        "reference_answer": (
            "Sí, un ensayo de trazador en el pozo inyector PI-08 (2024) mostró "
            "llegada al pozo productor PM-102 en solo 12 días (vs. 45 días "
            "históricos), confirmando canalización preferencial, con el corte "
            "de agua del PM-102 subiendo del 50% al 94%. Se decidió suspender "
            "la inyección y tratar con un tapón de geles."
        ),
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
        "reference_answer": (
            "Sí, en el pozo VM-210 (2023) el sistema BES se detuvo por baja "
            "aislación eléctrica (<0.1 Megaohms) detectada por el VSD. Al "
            "extraer la sarta se encontró el cable de potencia aplastado y "
            "quemado contra el casing a 2.100 metros, por falta de "
            "centralizadores en la zona desviada."
        ),
        "category": "BES",
    },
    {
        "id": "q7",
        "query": "¿A qué profundidad falló la tubería en el pozo LP-15?",
        "expected_keywords": ["tubing", "junta", "metros", "erosión", "LP-15"],
        "reference_answer": (
            'En el pozo LP-15 (2025) el tubing de 2 7/8" falló en la junta #54, '
            "a aproximadamente 1.620 metros de profundidad, con un orificio de "
            "1.5 pulgadas por erosión severa, justo frente a la válvula de Gas "
            "Lift número 3."
        ),
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
        "reference_answer": (
            "En el pozo PM-104 (2021), el equipo BES falló tras solo 45 días de "
            "Run Life por sobrecalentamiento del bobinado, con downthrust "
            "continuo causado por la declinación acelerada del aporte de "
            "fluido del reservorio, sin que el operador lo advirtiera a tiempo "
            "por falla del sensor de fondo."
        ),
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
        "reference_answer": (
            "El aumento de gas en el PM-104 se debe a una dinámica de "
            "reservorio: la presión del sector cayó a 1.650 psi, por debajo de "
            "la presión de burbuja de 1.850 psi, liberando gas disuelto y "
            "activando un mecanismo de empuje por gas, lo que incrementa el "
            "GOR, según el Informe de Estudios PVT del Yacimiento Los Perales "
            "(2024)."
        ),
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
        "reference_answer": (
            "La baja productividad crónica del pozo PM-108 no se debe a daño "
            "mecánico de la formación, sino a que el pozo interceptó una "
            "facies de llanura de inundación arcillosa (limolitas "
            "impermeables), quedando aislado del sistema principal de canales "
            "fluviales del yacimiento, según el estudio sedimentológico de la "
            "Formación Challacó."
        ),
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
        "reference_answer": (
            "En el pozo inyector PI-44 (2026) se detectó una falla en el "
            "packer (modelo Baker AD-1, a 1.950 metros) por comunicación "
            "directa entre el tubing de inyección y el espacio anular. La "
            "herramienta quedó trabada por incrustaciones de sulfato de bario "
            "sobre las gomas selladoras, y se liberó tras maniobras de "
            "tracción de hasta 30.000 lbs; el elemento sellador quedó "
            "destruido por extrusión química y térmica."
        ),
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
    semantic_similarity: float | None = None

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

                hits, score = _score_keywords(
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
                    keyword_total=len(q.get("expected_keywords", [])),
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
        best = max(
            report.summary.items(),
            key=lambda x: (x[1]["avg_keyword_score"], -x[1]["avg_elapsed_sec"]),
        )
        report.selected_model = best[0]
        stats = best[1]
        report.selection_rationale = (
            f"Mayor score de relevancia ({stats['avg_keyword_score']:.0%}) "
            f"+ similitud semántica {stats['avg_semantic_similarity']:.0%}) "
            f"con latencia de {stats['avg_elapsed_sec']:.1f}s promedio. "
            f"Mejor balance entre calidad de respuesta y rendimiento en CPU."
            f"Tasa de abstención: {stats['no_info_rate']:.0%}."
        )

    if output_path:
        report.save(output_path)

    report.print_summary()
    return report
