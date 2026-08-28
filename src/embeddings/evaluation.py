"""
Evaluación y comparación de modelos de embeddings candidatos.

Historia: "Pruebas de modelos candidatos"
  - evaluate_candidate_model(): genera embeddings de prueba, ejecuta las
    consultas de evaluación y registra los resultados de un modelo.
  - evaluate_models(): corre todos los candidatos y persiste resultados.

Historia: "Comparación y selección"
  - measure_quality(): comparación de calidad semántica.
  - evaluate_candidate_model(): también mide velocidad y consumo de RAM.
"""

from __future__ import annotations

import json
import logging
import time
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings.criteria import (
    APPROX_DISK_SIZE_MB,
    CANDIDATE_MODELS,
    EVALUATION_QUERIES,
)

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # medición de RAM es opcional, no bloquea el resto
    _HAS_PSUTIL = False

log = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResult:
    model_name: str
    ok: bool = True
    error: Optional[str] = None

    # Rendimiento
    embedding_dim: Optional[int] = None
    load_time_sec: Optional[float] = None
    encode_time_sec: Optional[float] = None
    texts_per_sec: Optional[float] = None

    # Recursos
    peak_ram_mb: Optional[float] = None
    approx_disk_size_mb: Optional[float] = None

    # Calidad semántica
    retrieval_accuracy: Optional[float] = None
    avg_margin: Optional[float] = None
    avg_similarity_correct: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _current_ram_mb() -> Optional[float]:
    if not _HAS_PSUTIL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def measure_quality(model: SentenceTransformer, evaluation_queries: list[dict]) -> dict:
    """
    Para cada consulta de evaluación, vectoriza query + doc correcto +
    doc incorrecto y verifica si el modelo asigna mayor similitud al
    documento correcto (acierto de retrieval).
    """
    if not evaluation_queries:
        return {
            "retrieval_accuracy": None,
            "avg_margin": None,
            "avg_similarity_correct": None,
        }

    aciertos = 0
    margenes = []
    similitudes_correctas = []

    for caso in evaluation_queries:
        vs = model.encode(
            [caso["query"], caso["doc_correcto"], caso["doc_incorrecto"]],
            convert_to_numpy=True,
        )
        sim_ok = float(cosine_similarity([vs[0]], [vs[1]])[0][0])
        sim_mal = float(cosine_similarity([vs[0]], [vs[2]])[0][0])

        if sim_ok > sim_mal:
            aciertos += 1
        margenes.append(sim_ok - sim_mal)
        similitudes_correctas.append(sim_ok)

    n = len(evaluation_queries)
    return {
        "retrieval_accuracy": round(aciertos / n, 3),
        "avg_margin": round(float(np.mean(margenes)), 4),
        "avg_similarity_correct": round(float(np.mean(similitudes_correctas)), 4),
    }


def evaluate_candidate_model(
    model_name: str,
    test_texts: list[str],
    evaluation_queries: list[dict],
    n_repeticiones: int = 5,
) -> ModelEvaluationResult:
    log.info("=== Evaluando modelo: %s ===", model_name)
    ram_antes = _current_ram_mb()

    try:
        t0 = time.perf_counter()
        model = SentenceTransformer(model_name)
        load_time = time.perf_counter() - t0

        # Warmup: no se mide, solo "calienta" el modelo (carga de kernels,
        # threads de PyTorch, cachés internas)
        model.encode(test_texts, convert_to_numpy=True)

        # Corridas medidas
        tiempos = []
        for _ in range(n_repeticiones):
            t0 = time.perf_counter()
            vectors = model.encode(test_texts, convert_to_numpy=True)
            tiempos.append(time.perf_counter() - t0)

        encode_time = statistics.median(tiempos)

        ram_despues = _current_ram_mb()
        peak_ram = (
            round(ram_despues - ram_antes, 1)
            if ram_antes is not None and ram_despues is not None
            else None
        )

        calidad = measure_quality(model, evaluation_queries)

        result = ModelEvaluationResult(
            model_name=model_name,
            embedding_dim=int(vectors.shape[1]),
            load_time_sec=round(load_time, 3),
            encode_time_sec=round(encode_time, 3),
            texts_per_sec=(
                round(len(test_texts) / encode_time, 1) if encode_time > 0 else None
            ),
            peak_ram_mb=peak_ram,
            approx_disk_size_mb=APPROX_DISK_SIZE_MB.get(model_name),
            **calidad,
        )
        log.info(
            "Resultado %s (mediana de %d corridas): dim=%s, %.1f textos/s, accuracy=%s, ram=%s MB",
            model_name,
            n_repeticiones,
            result.embedding_dim,
            result.texts_per_sec or 0,
            result.retrieval_accuracy,
            result.peak_ram_mb,
        )
        return result

    except Exception as e:
        log.exception("Error evaluando %s", model_name)
        return ModelEvaluationResult(model_name=model_name, ok=False, error=str(e))


def evaluate_models(
    test_texts: list[str],
    evaluation_queries: Optional[list[dict]] = None,
    candidates: Optional[list[str]] = None,
    results_path: Optional[str] = "evaluation_results.json",
) -> dict[str, ModelEvaluationResult]:
    """
    Evalúa todos los modelos candidatos sobre el mismo set de textos de
    prueba y consultas de evaluación, y persiste los resultados en JSON
    para trazabilidad (registro de resultados obtenidos).
    """
    evaluation_queries = (
        evaluation_queries if evaluation_queries is not None else EVALUATION_QUERIES
    )
    candidate_names = candidates or [c["name"] for c in CANDIDATE_MODELS]

    results: dict[str, ModelEvaluationResult] = {}
    for model_name in candidate_names:
        results[model_name] = evaluate_candidate_model(
            model_name, test_texts, evaluation_queries
        )

    if results_path:
        Path(results_path).write_text(
            json.dumps(
                {k: v.to_dict() for k, v in results.items()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("Resultados guardados en %s", results_path)

    return results


def generate_report(
    results: dict[str, ModelEvaluationResult],
    accuracy_threshold: float = 0.75,
) -> str:
    """
    Genera un reporte en Markdown con la tabla comparativa y una
    justificación automática basada en los umbrales definidos en
    criteria.COMPARISON_CRITERIA.
    """
    lines = ["# Evaluación de modelos de embeddings — resultados\n"]
    lines.append(
        "| Modelo | Dim | Textos/s | RAM (MB) | Disco aprox (MB) | Accuracy | Margen prom. |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for name, r in results.items():
        if not r.ok:
            lines.append(f"| {name} | ERROR: {r.error} | | | | | |")
            continue
        lines.append(
            f"| {name} | {r.embedding_dim} | {r.texts_per_sec} | "
            f"{r.peak_ram_mb} | {r.approx_disk_size_mb} | "
            f"{r.retrieval_accuracy} | {r.avg_margin} |"
        )

    validos = {
        k: v
        for k, v in results.items()
        if v.ok
        and v.retrieval_accuracy is not None
        and v.retrieval_accuracy >= accuracy_threshold
    }

    lines.append("\n## Justificación\n")
    if not validos:
        lines.append(
            f"Ningún modelo alcanzó el umbral mínimo de calidad "
            f"(retrieval_accuracy >= {accuracy_threshold}). Se recomienda "
            f"ampliar el set de EVALUATION_QUERIES antes de decidir, ya que "
            f"con pocos casos el resultado puede no ser representativo."
        )
    else:
        seleccionado = max(validos.items(), key=lambda kv: kv[1].texts_per_sec or 0)
        nombre, r = seleccionado
        lines.append(
            f"**Modelo seleccionado: `{nombre}`**\n\n"
            f"- Cumple el umbral de calidad definido "
            f"(accuracy={r.retrieval_accuracy} >= {accuracy_threshold}).\n"
            f"- Entre los modelos que cumplen el umbral, es el de mayor "
            f"velocidad de generación ({r.texts_per_sec} textos/s en CPU).\n"
            f"- Consumo de RAM medido: {r.peak_ram_mb} MB. "
            f"Tamaño en disco aproximado: {r.approx_disk_size_mb} MB.\n"
        )
        otros = [k for k in validos if k != nombre]
        if otros:
            lines.append(
                f"- Otros modelos que también cumplieron el umbral de calidad "
                f"({', '.join(otros)}) fueron descartados por menor velocidad "
                f"y/o mayor consumo de recursos, sin ofrecer una mejora de "
                f"calidad que lo justifique."
            )

    return "\n".join(lines)
