"""Selección del mejor modelo."""

import math

LATENCY_PENALTY_PER_DOUBLING = 0.05


def _select_best_model(
    summary: dict[str, dict[str, float]],
    latency_penalty: float = LATENCY_PENALTY_PER_DOUBLING,
    min_faithfulness: float | None = None,
) -> tuple[str, str]:
    """Elige el modelo que mejor responde sin aceptar una latencia desproporcionada."""
    if not summary:
        return "", ""

    candidates: dict[str, dict[str, float]] = {}
    discarded: list[str] = []
    for model, s in summary.items():
        if s["ok_runs"] == 0:
            discarded.append(f"{model} (sin corridas exitosas)")
        elif (
            min_faithfulness is not None
            and 0 < s["avg_judge_faithfulness"] < min_faithfulness
        ):
            discarded.append(
                f"{model} (fidelidad {s['avg_judge_faithfulness']:.1f} < {min_faithfulness})"
            )
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
    parts = [
        (
            f"Calidad {s['quality']:.0%} con latencia máxima (p95) de {s['p95_elapsed_sec']:.1f}s "
            f"(típica p50 {s['p50_elapsed_sec']:.1f}s): es el mejor equilibrio entre calidad y tiempo de respuesta."
        )
    ]
    if len(candidates) > 1:
        ranking = sorted(candidates, key=_adjusted, reverse=True)
        parts.append(
            "Comparación entre modelos: "
            + ", ".join(
                f"{m} (calidad {candidates[m]['quality']:.0%}, p95 {candidates[m]['p95_elapsed_sec']:.1f}s)"
                for m in ranking
            )
            + f". Cada vez que un modelo tarda el doble, se le exigen {latency_penalty * 100:.0f} puntos más de calidad."
        )

    parts.append(
        f"Keywords {s['avg_keyword_score']:.0%}, "
        f"similitud semántica {s['avg_semantic_similarity']:.0%}."
    )
    if s["total_negative"]:
        parts.append(
            f"Alucinación en preguntas sin respuesta: {s['hallucination_rate']:.0%}."
        )
    parts.append(f"Abstención incorrecta: {s['false_abstention_rate']:.0%}.")
    if discarded:
        parts.append("Descartados: " + "; ".join(discarded) + ".")
    return best, " ".join(parts)
