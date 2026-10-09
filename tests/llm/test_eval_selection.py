"""
Cubre "Mejorar precisión del scoring en la evaluación de LLMs: el matching exacto de keywords
subestimaba la calidad real de las respuestas":
  - Selección del modelo con criterio combinado -> CA-19.3

Cubre "Mejorar la selección de modelos LLM en la evaluación RAG: scoring más robusto, medición de latencia confiable y elección que combina calidad y tiempo de respuesta"
  - Selección del modelo -> CA-28.1, CA-28.2, CA-28.3, CA-28.4, 28.6, 28.7 y 28.8
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

    # CA-28.2: El sistema debe comparar automáticamente los tiempos de respuesta
    # entre los modelos evaluados, dando prioridad al tiempo en los casos más lentos,
    # sin que haya que definir un límite de tiempo.
    def test_rationale_compares_latency_between_models(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "modelo_lento": _stats(quality=0.80, p95_elapsed_sec=60.0),
            "modelo_rapido": _stats(quality=0.80, p95_elapsed_sec=10.0),
        }
        _, rationale = _select_best_model(summary)

        assert "modelo_lento (calidad 80%, p95 60.0s)" in rationale
        assert "modelo_rapido (calidad 80%, p95 10.0s)" in rationale

    # CA-28.2: El sistema debe comparar automáticamente los tiempos de respuesta
    # entre los modelos evaluados, dando prioridad al tiempo en los casos más lentos,
    # sin que haya que definir un límite de tiempo.
    def test_single_model_has_no_comparison(self):
        from src.llm.evaluator import _select_best_model

        _, rationale = _select_best_model({"A": _stats()})
        assert "Comparación entre modelos" not in rationale

    # CA-28.3: Si se define una fidelidad mínima,
    # el sistema debe descartar los modelos que no la alcancen,
    # siempre que haya evaluación del juez.
    def test_min_faithfulness_only_applies_when_judged(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.9, avg_judge_faithfulness=3.0),
            "B": _stats(quality=0.8, avg_judge_faithfulness=0.0),  # sin juez
        }
        assert _select_best_model(summary, min_faithfulness=4.0)[0] == "B"

    # CA-28.3: Si se define una fidelidad mínima,
    # el sistema debe descartar los modelos que no la alcancen,
    # siempre que haya evaluación del juez.
    def test_model_meeting_min_faithfulness_is_kept(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.9, avg_judge_faithfulness=4.5),
            "B": _stats(quality=0.8, avg_judge_faithfulness=5.0),
        }
        assert _select_best_model(summary, min_faithfulness=4.0)[0] == "A"

    # CA-28.4: El sistema debe descartar los modelos que no lograron ejecutar
    # ninguna consulta con éxito.
    def test_excludes_models_without_successful_runs(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(
                quality=0.0, p50_elapsed_sec=0.0, p95_elapsed_sec=0.0, ok_runs=0.0
            ),
            "B": _stats(quality=0.0, p50_elapsed_sec=9.0, p95_elapsed_sec=9.0),
        }
        assert _select_best_model(summary)[0] == "B"

    # CA-28.4: El sistema debe descartar los modelos que no lograron ejecutar
    # ninguna consulta con éxito.
    def test_no_model_selected_when_none_has_successful_runs(self):
        from src.llm.evaluator import _select_best_model

        best, rationale = _select_best_model({"A": _stats(ok_runs=0.0)})
        assert best == ""
        assert "Ningún modelo" in rationale

    # CA-28.7: La justificación debe reportar la calidad, los tiempos de respuesta,
    # la abstención incorrecta y la alucinación del modelo elegido,
    # y mencionar los modelos descartados.
    def test_rationale_reports_quality_latency_and_abstention(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "modelo_a": _stats(),
            "modelo_b": _stats(avg_judge_faithfulness=2.0),
        }
        _, rationale = _select_best_model(summary, min_faithfulness=4.0)

        for text in (
            "Calidad",
            "p50",
            "Abstención incorrecta",
            "Alucinación",
            "Descartados",
            "modelo_b",
        ):
            assert text in rationale

    # CA-28.7 La justificación debe reportar la calidad, los tiempos de respuesta,
    # la abstención incorrecta, el score de keywords, la similitud semántica
    # y la alucinación del modelo elegido, y mencionar los modelos descartados.
    def test_rationale_omits_hallucination_without_negatives(self):
        from src.llm.evaluator import _select_best_model

        _, rationale = _select_best_model({"A": _stats(total_negative=0.0)})
        assert "Alucinación" not in rationale

    # CA-28.7 La justificación debe reportar la calidad, los tiempos de respuesta,
    # la abstención incorrecta, el score de keywords, la similitud semántica
    # y la alucinación del modelo elegido, y mencionar los modelos descartados.
    def test_rationale_reports_quality_and_abstention_values(self):
        from src.llm.evaluator import _select_best_model

        _, rationale = _select_best_model(
            {
                "A": _stats(
                    quality=0.8, avg_keyword_score=0.7, avg_semantic_similarity=0.9
                )
            }
        )
        assert "80%" in rationale
        assert "Abstención incorrecta: 0%" in rationale
        assert "Keywords 70%" in rationale
        assert "similitud semántica 90%" in rationale


class TestSelectionLimits:
    # CA-28.9: Se descarta el modelo cuya tasa de abstención incorrecta supere el máximo.
    def test_discards_high_false_abstention(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.9, false_abstention_rate=0.2),
            "B": _stats(quality=0.7),
        }
        assert _select_best_model(summary)[0] == "B"

    # CA-28.9: Se descarta el modelo cuya tasa de abstención incorrecta supere el máximo.
    def test_false_abstention_limit_can_be_disabled(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.9, false_abstention_rate=0.2),
            "B": _stats(quality=0.7),
        }
        assert _select_best_model(summary, max_false_abstention=None)[0] == "A"

    # CA-28.10
    def test_discards_high_hallucination(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.9, hallucination_rate=0.5),
            "B": _stats(quality=0.7),
        }
        assert _select_best_model(summary)[0] == "B"

    # CA-28.10
    def test_hallucination_limit_ignored_without_negatives(self):
        from src.llm.evaluator import _select_best_model

        summary = {"A": _stats(hallucination_rate=1.0, total_negative=0.0)}
        assert _select_best_model(summary)[0] == "A"

    # CA-28.10
    def test_hallucination_limit_can_be_disabled(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "A": _stats(quality=0.9, hallucination_rate=0.5),
            "B": _stats(quality=0.7),
        }
        assert _select_best_model(summary, max_hallucination=None)[0] == "A"
