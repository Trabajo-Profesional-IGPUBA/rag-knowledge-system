"""
Cubre "Mejorar precisión del scoring en la evaluación de LLMs: el matching exacto de keywords
subestimaba la calidad real de las respuestas":
  - Selección del modelo con criterio combinado -> CA-19.3

Cubre "Mejorar la selección de modelos LLM en la evaluación RAG: scoring más robusto, medición de latencia confiable y elección que combina calidad y tiempo de respuesta"
  - Selección del modelo -> CA-28.1, CA-28.2, 28.6 y 28.8
"""

from tests.llm.helpers import (
    _stats,
)


class TestSelectionCriteria:
    # CA-28.1: El sistema debe elegir el modelo que mejor responde,
    # pero sin aceptar una latencia poco razonable: un modelo notablemente más lento
    # que los demás solo debe ser elegido si su ventaja en calidad lo compensa.
    def test_selects_model_with_best_quality(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "modelo_a": _stats(quality=0.3, p50_elapsed_sec=10.0),
            "modelo_b": _stats(quality=0.8, p50_elapsed_sec=20.0),
        }
        assert _select_best_model(summary)[0] == "modelo_b"

    # CA-28.8: Si dos modelos tienen la misma calidad,
    # el sistema debe elegir el más rápido.
    def test_same_quality_picks_the_fastest(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.80, p50_elapsed_sec=20.0, p95_elapsed_sec=30.0),
            "B": _stats(quality=0.80, p50_elapsed_sec=5.0, p95_elapsed_sec=10.0),
        }
        assert _select_best_model(summary)[0] == "B"

    # CA-19.3: Si no hay ningún modelo evaluado (resumen vacío), el sistema no debe fallar al intentar seleccionar el mejor modelo,
    # y debe devolver un modelo seleccionado y una justificación vacíos.
    # CA-28.6: Si no hay ningún modelo evaluado, el sistema no debe fallar
    # y debe devolver un modelo seleccionado y una justificación vacíos (mantiene CA-19.3).
    def test_empty_summary_returns_empty_selection(self):
        from src.llm.evaluator import _select_best_model

        assert _select_best_model({}) == ("", "")


class TestModelSelection:
    # CA-28.1 El sistema debe elegir el modelo que mejor responde,
    # pero sin aceptar una latencia poco razonable: un modelo notablemente más lento
    # que los demás solo debe ser elegido si su ventaja en calidad lo compensa.
    def test_best_quality_wins_when_latency_is_comparable(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.80, p50_elapsed_sec=10.0, p95_elapsed_sec=10.0),
            "B": _stats(quality=0.75, p50_elapsed_sec=9.0, p95_elapsed_sec=9.0),
        }
        assert _select_best_model(summary)[0] == "A"

    # CA-28.1 El sistema debe elegir el modelo que mejor responde,
    # pero sin aceptar una latencia poco razonable: un modelo notablemente más lento
    # que los demás solo debe ser elegido si su ventaja en calidad lo compensa.
    def test_much_slower_model_needs_a_bigger_quality_advantage(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "lento_algo_mejor": _stats(
                quality=0.80, p50_elapsed_sec=40.0, p95_elapsed_sec=40.0
            ),
            "rapido": _stats(quality=0.76, p50_elapsed_sec=10.0, p95_elapsed_sec=10.0),
        }
        assert _select_best_model(summary)[0] == "rapido"

    # CA-28.1 El sistema debe elegir el modelo que mejor responde,
    # pero sin aceptar una latencia poco razonable: un modelo notablemente más lento
    # que los demás solo debe ser elegido si su ventaja en calidad lo compensa.
    def test_slower_model_wins_when_quality_advantage_compensates(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "lento_mucho_mejor": _stats(
                quality=0.90, p50_elapsed_sec=40.0, p95_elapsed_sec=40.0
            ),
            "rapido": _stats(quality=0.76, p50_elapsed_sec=10.0, p95_elapsed_sec=10.0),
        }
        assert _select_best_model(summary)[0] == "lento_mucho_mejor"

    # CA-28.2: El sistema debe comparar automáticamente los tiempos de respuesta
    # entre los modelos evaluados, dando prioridad al tiempo en los casos más lentos,
    # sin que haya que definir un límite de tiempo.
    def test_prioritizes_latency_in_slowest_cases(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "modelo_picos": _stats(
                quality=0.80, p50_elapsed_sec=2.0, p95_elapsed_sec=60.0
            ),
            "modelo_estable": _stats(
                quality=0.80, p50_elapsed_sec=5.0, p95_elapsed_sec=10.0
            ),
        }
        assert _select_best_model(summary)[0] == "modelo_estable"
