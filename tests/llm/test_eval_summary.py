import pytest

from tests.llm.helpers import (
    _result,
)


class TestSummarySemanticSimilarity:
    # CA-18.3: El resumen por modelo debe reportar el promedio de similitud
    # semántica sobre las consultas donde pudo calcularse.
    def test_summary_averages_only_available_similarity_values(self):
        from src.llm.eval_summary import _summarize

        results = [_result(semantic_similarity=0.9), _result(semantic_similarity=None)]
        assert _summarize(results)["avg_semantic_similarity"] == 0.9


class TestErrorPenalty:
    # CA-22.1: Una consulta que falla debe contar como calidad cero para el modelo,
    # en lugar de quedar fuera del promedio.

    def test_errors_count_as_zero_quality(self):
        from src.llm.eval_summary import _summarize

        results = [
            _result(quality=1.0),
            _result(quality=0.5),
            _result(quality=0.0, error="boom", elapsed_sec=0.0),
        ]
        assert _summarize(results)["quality"] == pytest.approx(0.5)

    # CA-22.2: El resumen por modelo debe reportar la cantidad de errores
    # y el total de consultas ejecutadas.

    def test_summary_reports_error_count_and_total_runs(self):
        from src.llm.eval_summary import _summarize

        summary = _summarize([_result(), _result(error="boom")])
        assert summary["error_count"] == 1
        assert summary["total_runs"] == 2
        assert summary["ok_runs"] == 1


class TestSummaryNoInfoRate:
    # CA-25.3: Ante las preguntas que sí tienen respuesta,
    # el resumen debe reportar la tasa de abstención incorrecta,
    # calculada solo sobre las consultas ejecutadas sin error.
    def test_summary_includes_false_abstention_rate(self):
        from src.llm.eval_summary import _summarize

        results = [
            _result(abstained=True, keyword_total=3),
            _result(abstained=False, keyword_total=3),
            _result(abstained=False, error="timeout"),  # no debe contarse
        ]
        assert _summarize(results)["false_abstention_rate"] == 0.5


class TestSummaryJudge:
    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_summary_averages_only_judged_runs(self):
        from src.llm.eval_summary import _summarize

        summary = _summarize(
            [
                _result(judge_correctness=5.0, judge_faithfulness=4.0),
                _result(judge_correctness=3.0, judge_faithfulness=2.0),
                _result(),  # sin juez: no cuenta
            ]
        )
        assert summary["avg_judge_correctness"] == 4.0
        assert summary["avg_judge_faithfulness"] == 3.0

    # CA-24.3: Si no se indica un modelo juez, el sistema debe evaluar igual,
    # detectando la abstención por el texto de la respuesta.
    def test_summary_without_judge_is_zero(self):
        from src.llm.eval_summary import _summarize

        summary = _summarize([_result()])
        assert summary["avg_judge_correctness"] == 0.0
        assert summary["avg_judge_faithfulness"] == 0.0

    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_print_summary_shows_judge_scores(self, capsys):
        from src.llm.eval_models import EvaluationReport
        from src.llm.eval_summary import _summarize

        report = EvaluationReport(models_evaluated=["m"])
        report.summary["m"] = _summarize(
            [_result(judge_correctness=4.0, judge_faithfulness=5.0)]
        )
        report.print_summary()

        assert "4.0 / 5.0 (de 5)" in capsys.readouterr().out

    # CA-25.5
    def test_avg_keyword_score_ignores_runs_without_answer(self):
        from src.llm.evaluator import _summarize

        summary = _summarize(
            [
                _result(keyword_total=2, keyword_hits=2, keyword_score=1.0),
                _result(should_abstain=True),  # negativa: no debe bajar el promedio
            ]
        )
        assert summary["avg_keyword_score"] == 1.0

    # CA-22.3
    def test_avg_keyword_score_counts_failed_runs_as_zero(self):
        from src.llm.evaluator import _summarize

        summary = _summarize(
            [
                _result(keyword_total=2, keyword_hits=2, keyword_score=1.0),
                _result(error="boom"),
            ]
        )
        assert summary["avg_keyword_score"] == 0.5


class TestLatencyMeasurement:
    # CA-27.2: El resumen debe reportar el tiempo típico, el promedio y el tiempo
    # en los casos más lentos, calculados solo sobre las consultas ejecutadas sin error.
    def test_latency_stats_only_use_successful_runs(self):
        from src.llm.eval_summary import _summarize

        summary = _summarize(
            [
                _result(elapsed_sec=1.0),
                _result(elapsed_sec=3.0),
                _result(elapsed_sec=0.0, error="boom"),
            ]
        )
        assert summary["p50_elapsed_sec"] == 2.0
        assert summary["avg_elapsed_sec"] == 2.0

    # CA-27.2: El resumen debe reportar el tiempo típico, el promedio y el tiempo
    # en los casos más lentos, calculados solo sobre las consultas ejecutadas sin error.
    def test_slowest_cases_latency(self):
        from src.llm.eval_summary import _summarize

        results = [_result(elapsed_sec=float(i)) for i in range(1, 21)]
        assert _summarize(results)["p95_elapsed_sec"] == 19.0


class TestRetrieverSeparation:
    # CA-26.3: El resumen debe reportar la calidad del modelo
    # excluyendo las consultas donde se comprobó que la información no contenía los datos,
    # y la efectividad del buscador por separado.
    def test_summary_separates_quality_by_context(self):
        from src.llm.eval_summary import _summarize

        summary = _summarize(
            [
                _result(quality=1.0, context_ok=True, context_recall=1.0),
                _result(quality=0.0, context_ok=False, context_recall=0.0),
            ]
        )
        assert summary["quality_ctx_ok"] == 1.0
        assert summary["quality"] == 0.5
        assert summary["retrieval_recall"] == 0.5
