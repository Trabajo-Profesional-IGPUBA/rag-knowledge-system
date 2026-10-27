"""Resumen por modelo."""

import math
import statistics

from src.llm.eval_models import ModelEvalResult

# Decimales con que se redondean las métricas de calidad y tasas (0-1).
METRIC_DECIMALS = 4

# Decimales con que se redondean los tiempos (segundos).
LATENCY_DECIMALS = 2

# Percentil que se reporta como latencia de cola (p95).
TAIL_LATENCY_PERCENTILE = 0.95


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), METRIC_DECIMALS) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Percentil por nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1))
    return round(ordered[idx], LATENCY_DECIMALS)


def _summarize(results: list[ModelEvalResult]) -> dict[str, float]:
    """Resumen agregado de todas las corridas de un modelo."""
    answerable = [r for r in results if not r.should_abstain]
    negatives = [r for r in results if r.should_abstain]
    ok = [r for r in results if r.ok]
    ok_ans = [r for r in answerable if r.ok]
    ok_neg = [r for r in negatives if r.ok]
    latencies = [r.elapsed_sec for r in ok]

    return {
        "quality": _mean([r.quality for r in results]),  # errores cuentan como 0
        "quality_ctx_ok": _mean(
            [r.quality for r in answerable if r.context_ok is not False]
        ),
        "avg_keyword_score": _mean(
            [r.keyword_score for r in answerable if r.ok and r.keyword_total > 0]
            + [0.0 for r in answerable if not r.ok]
        ),
        "avg_number_score": _mean(
            [r.number_score for r in answerable if r.number_score is not None]
        ),
        "avg_semantic_similarity": _mean(
            [r.semantic_similarity for r in ok_ans if r.semantic_similarity is not None]
        ),
        "avg_judge_correctness": _mean(
            [r.judge_correctness for r in ok if r.judge_correctness is not None]
        ),
        "avg_judge_faithfulness": _mean(
            [r.judge_faithfulness for r in ok if r.judge_faithfulness is not None]
        ),
        "retrieval_recall": _mean(
            [r.context_recall for r in answerable if r.context_recall is not None]
        ),
        "false_abstention_rate": _mean([1.0 if r.abstained else 0.0 for r in ok_ans]),
        "hallucination_rate": _mean([0.0 if r.abstained else 1.0 for r in ok_neg]),
        "avg_elapsed_sec": (
            round(statistics.fmean(latencies), LATENCY_DECIMALS) if latencies else 0.0
        ),
        "p50_elapsed_sec": (
            round(statistics.median(latencies), LATENCY_DECIMALS) if latencies else 0.0
        ),
        "p95_elapsed_sec": _percentile(latencies, TAIL_LATENCY_PERCENTILE),
        "avg_response_length": _mean([float(r.response_length) for r in ok]),
        "error_count": float(len([r for r in results if not r.ok])),
        "ok_runs": float(len(ok)),
        "total_runs": float(len(results)),
        "total_negative": float(len(negatives)),
    }
