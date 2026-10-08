"""
Cubre "Mejorar la selección de modelos LLM en la evaluación RAG: scoring más robusto, medición de latencia confiable y elección que combina calidad y tiempo de respuesta"
  - Errores durante la evaluación-> CA-22.1
  - Calidad de cada respuesta-> CA-23.1 a 23.2
"""

import pytest

from tests.llm.helpers import (
    _result,
)


class TestRunQuality:
    # CA-22.1: Una consulta que falla debe contar como calidad cero para el modelo,
    # en lugar de quedar fuera del promedio.
    def test_failed_run_has_zero_quality(self):
        from src.llm.evaluator import _run_quality

        r = _result(error="boom", keyword_total=2, keyword_score=1.0)
        assert _run_quality(r) == 0.0

    # CA-23.1: El sistema debe calcular la calidad de cada respuesta combinando
    # las medidas disponibles (evaluación del juez, palabras clave, cifras
    # y similitud semántica), usando solo las que se pudieron obtener.
    def test_uses_only_available_components(self):
        from src.llm.evaluator import _run_quality

        r = _result(keyword_total=2, keyword_hits=1, keyword_score=0.5)
        assert _run_quality(r) == pytest.approx(0.5)

    # CA-23.1: El sistema debe calcular la calidad de cada respuesta combinando
    # las medidas disponibles (evaluación del juez, palabras clave, cifras
    # y similitud semántica), usando solo las que se pudieron obtener.
    def test_judge_score_is_normalized(self):
        from src.llm.evaluator import _run_quality

        assert _run_quality(_result(judge_correctness=5.0)) == pytest.approx(1.0)
        assert _run_quality(_result(judge_correctness=1.0)) == pytest.approx(0.0)

    # CA-23.1: El sistema debe calcular la calidad de cada respuesta combinando
    # las medidas disponibles (evaluación del juez, palabras clave, cifras
    # y similitud semántica), usando solo las que se pudieron obtener.
    def test_number_score_is_used(self):
        from src.llm.evaluator import _run_quality

        assert _run_quality(_result(number_score=0.5)) == pytest.approx(0.5)

    # CA-23.1: El sistema debe calcular la calidad de cada respuesta combinando
    # las medidas disponibles (evaluación del juez, palabras clave, cifras
    # y similitud semántica), usando solo las que se pudieron obtener.
    def test_semantic_similarity_is_rescaled(self):
        from src.llm.evaluator import _run_quality

        assert _run_quality(_result(semantic_similarity=0.75)) == pytest.approx(0.5)
        assert _run_quality(_result(semantic_similarity=0.4)) == 0.0

    # CA-23.2: La similitud semántica debe pesar menos que las demás medidas,
    # porque distingue poco entre respuestas buenas y malas sobre el mismo tema.
    def test_semantic_weight_is_lowest(self):
        from src.llm.evaluator import QUALITY_WEIGHTS

        assert QUALITY_WEIGHTS["semantic"] == min(QUALITY_WEIGHTS.values())
