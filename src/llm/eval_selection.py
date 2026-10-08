"""Selección del mejor modelo."""

import math

LATENCY_PENALTY_PER_DOUBLING = 0.05


def _select_best_model(
    summary: dict[str, dict[str, float]],
    latency_penalty: float = LATENCY_PENALTY_PER_DOUBLING,
) -> tuple[str, str]:
    """Elige el modelo que mejor responde sin aceptar una latencia desproporcionada."""
    if not summary:
        return "", ""

    def _latency(m: str) -> float:
        return max(summary[m]["p95_elapsed_sec"], 0.01)

    fastest = min(_latency(m) for m in summary)

    def _adjusted(m: str) -> float:
        return summary[m]["quality"] - latency_penalty * math.log2(
            _latency(m) / fastest
        )

    best = max(
        summary,
        key=lambda m: (
            round(_adjusted(m), 6),
            -_latency(m),
            -summary[m]["p50_elapsed_sec"],
        ),
    )

    s = summary[best]
    rationale = (
        f"Calidad {s['quality']:.0%} con latencia máxima (p95) de "
        f"{s['p95_elapsed_sec']:.1f}s (típica p50 {s['p50_elapsed_sec']:.1f}s): "
        "es el mejor equilibrio entre calidad y tiempo de respuesta."
    )
    return best, rationale
