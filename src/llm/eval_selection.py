"""Selección del mejor modelo."""

import math

# Cada vez que un modelo tarda el doble que el más rápido, se le exige esta ventaja
# (en puntos de calidad, 0-1) para ser elegido. Es una comparación relativa entre modelos.
LATENCY_PENALTY_PER_DOUBLING = 0.05

# máx. % de veces que puede negarse a responder cuando sí había info
MAX_FALSE_ABSTENTION = 0.05
# máx. % de veces que puede inventar cuando NO había info
MAX_HALLUCINATION = 0.10


def _select_best_model(
    summary: dict[str, dict[str, float]],
    latency_penalty: float = LATENCY_PENALTY_PER_DOUBLING,
    min_faithfulness: float | None = None,
    max_false_abstention: float | None = MAX_FALSE_ABSTENTION,
    max_hallucination: float | None = MAX_HALLUCINATION,
) -> tuple[str, str]:
    """Elige el modelo que mejor responde sin aceptar una latencia desproporcionada.

    1. Descarta modelos sin corridas exitosas o con fidelidad (judge, 1-5) < `min_faithfulness`.
    2. Compara la latencia entre los modelos restantes (p95, casos más lentos): cada vez que
       un modelo tarda el doble que el más rápido, se le restan `latency_penalty` puntos de calidad.
    3. Gana el mayor puntaje ajustado; a igualdad, el más rápido. No requiere ningún límite externo.
    """

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
        elif (
            max_false_abstention is not None
            and s["false_abstention_rate"] > max_false_abstention
        ):
            discarded.append(
                f"{model} (abstención incorrecta {s['false_abstention_rate']:.0%} > {max_false_abstention:.0%})"
            )
        elif (
            max_hallucination is not None
            and s["total_negative"]
            and s["hallucination_rate"] > max_hallucination
        ):
            discarded.append(
                f"{model} (alucinación {s['hallucination_rate']:.0%} > {max_hallucination:.0%})"
            )
        else:
            candidates[model] = s

    if not candidates:

        def _fails(m: str) -> tuple:
            s = summary[m]
            return (s["hallucination_rate"] + s["false_abstention_rate"], -s["quality"])

        viable = [m for m in summary if summary[m]["ok_runs"] > 0]
        if not viable:
            return "", "Ningún modelo tuvo corridas exitosas."
        best = min(viable, key=_fails)
        return best, (
            "Ningún modelo cumplió los límites de abstención/alucinación; "
            f"se eligió {best} por tener la menor suma de fallos. "
            "Descartados: " + "; ".join(discarded) + "."
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
