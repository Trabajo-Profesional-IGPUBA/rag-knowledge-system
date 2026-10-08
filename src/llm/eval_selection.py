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

    candidates: dict[str, dict[str, float]] = {}
    discarded: list[str] = []
    for model, s in summary.items():
        if s["ok_runs"] == 0:
            discarded.append(f"{model} (sin corridas exitosas)")
        else:
            candidates[model] = s

    if not candidates:
        return "", (
            "Ningún modelo cumplió los requisitos. Descartados: "
            + "; ".join(discarded)
            + "."
        )

    def _latency(m: str) -> float:
        return max(candidates[m]["p95_elapsed_sec"], 0.01)

    fastest = min(_latency(m) for m in candidates)

    def _adjusted(m: str) -> float:
        return candidates[m]["quality"] - latency_penalty * math.log2(
            _latency(m) / fastest
        )

    best = max(
        candidates,
        key=lambda m: (
            round(_adjusted(m), 6),
            -_latency(m),
            -candidates[m]["p50_elapsed_sec"],
        ),
    )

    s = candidates[best]
    rationale = (
        f"Calidad {s['quality']:.0%} con latencia máxima (p95) de "
        f"{s['p95_elapsed_sec']:.1f}s (típica p50 {s['p50_elapsed_sec']:.1f}s): "
        "es el mejor equilibrio entre calidad y tiempo de respuesta."
    )
    return best, rationale
