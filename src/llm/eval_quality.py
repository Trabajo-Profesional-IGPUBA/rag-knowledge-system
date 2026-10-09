"""Calidad de una corrida."""

from src.llm.eval_models import ModelEvalResult

# Pesos de cada componente en la calidad de una respuesta (se renormalizan
# si algún componente no está disponible, p. ej. sin judge).
QUALITY_WEIGHTS = {"judge": 0.40, "keywords": 0.25, "numbers": 0.25, "semantic": 0.10}

# Escala de puntaje del juez; debe coincidir con la que se le pide en el prompt.
JUDGE_MIN_SCORE = 1
JUDGE_MAX_SCORE = 5

# Similitud coseno a partir de la cual se empieza a dar crédito semántico.
# Por debajo (o igual) vale 0; el coseno crudo casi no separa respuestas.
SEMANTIC_FLOOR = 0.5


def _run_quality(r: ModelEvalResult) -> float:
    """Calidad 0-1 de una corrida. Error = 0. Negativas: 1 si se abstuvo, 0 si no."""
    if not r.ok:
        return 0.0
    if r.should_abstain:
        return 1.0 if r.abstained else 0.0

    components: dict[str, float] = {}
    if r.judge_correctness is not None:
        components["judge"] = (r.judge_correctness - JUDGE_MIN_SCORE) / (
            JUDGE_MAX_SCORE - JUDGE_MIN_SCORE
        )
    if r.keyword_total > 0:
        components["keywords"] = r.keyword_score
    if r.number_score is not None:
        components["numbers"] = r.number_score
    if r.semantic_similarity is not None:
        # reescala: SEMANTIC_FLOOR o menos → 0, 1.0 → 1
        components["semantic"] = max(
            0.0, (r.semantic_similarity - SEMANTIC_FLOOR) / (1.0 - SEMANTIC_FLOOR)
        )

    if not components:
        return 0.0
    weight_sum = sum(QUALITY_WEIGHTS[k] for k in components)
    return sum(QUALITY_WEIGHTS[k] * v for k, v in components.items()) / weight_sum
