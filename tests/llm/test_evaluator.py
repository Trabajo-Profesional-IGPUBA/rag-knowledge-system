"""
Cubre "ÉPICA: Cliente LLM sobre Ollama":
  - Ejecución de la evaluación -> CA-12.1 a CA-12.4
  - Manejo de errores durante la evaluación -> CA-13.1 a CA-13.2
  - Selección del modelo -> CA-14.1 a CA-14.2
  - Persistencia de resultados -> CA-15.1

Cubre "Mejorar precisión del scoring en la evaluación de LLMs: el matching exacto de keywords
subestimaba la calidad real de las respuestas":
  - Detección de abstención -> 17.2
  - Similitud semántica-> CA-18.4
  - Selección del modelo con criterio combinado -> CA-19.3

Cubre "Mejorar la selección de modelos LLM en la evaluación RAG: scoring más robusto, medición de latencia confiable y elección que combina calidad y tiempo de respuesta"
  - Errores durante la evaluación-> CA-22.1
  - Evaluación por un modelo juez-> CA-24.1 a CA-24.4
  - Consultas sin respuesta-> CA-25.2, CA-25.3 y 25.4
  - Información recuperada por el buscador-> CA-26.1 a 26.2
  - Tiempos de respuesta-> CA-27.1, 27.3 y 27.4
  - Robustez de la ejecución-> CA-29.1 a CA-29.3
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.llm.helpers import (
    ANSWERS,
    QUERIES,
    _mock_client,
    _pipeline_by_query,
    _rag_resp,
    _result,
)


class TestEvaluateModels:
    def _make_mock_pipeline(
        self, answer="La presión es 3500 psi", raise_on_query=False
    ):
        pipeline = MagicMock()
        if raise_on_query:
            pipeline.query.side_effect = RuntimeError("fallo de conexión")
        else:
            pipeline.query.return_value = _rag_resp(answer)
        return pipeline

    # CA-12.1: El sistema debe evaluar cada modelo indicado contra el conjunto de preguntas, registrando la respuesta, el tiempo de ejecución y el score de palabras clave obtenidos.
    # CA-12.4: Si un modelo ya está descargado, el sistema no debe intentar descargarlo de nuevo.
    @patch("src.llm.evaluator.RAGPipeline")
    @patch("src.llm.evaluator.PromptBuilder")
    @patch("src.llm.evaluator.LLMClient")
    def test_evaluate_models_happy_path(
        self, mock_llm_client_cls, mock_prompt_builder_cls, mock_rag_pipeline_cls
    ):
        from src.llm.evaluator import evaluate_models

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.list_models.return_value = ["llama3:8b"]
        mock_llm_client_cls.return_value = mock_client

        mock_rag_pipeline_cls.return_value = self._make_mock_pipeline(
            answer="La presión de fondo del pozo PM-104 es 3500 psi"
        )

        retriever = MagicMock()
        queries = [
            {
                "id": "q1",
                "query": "¿Cuál es la presión de fondo del pozo PM-104?",
                "expected_keywords": ["presión", "psi", "PM-104"],
            }
        ]

        report = evaluate_models(
            retriever=retriever,
            models=["llama3:8b"],
            queries=queries,
            n_runs=1,
        )

        assert report.selected_model == "llama3:8b"
        assert report.results[0].keyword_score == 1.0
        assert report.results[0].error is None
        mock_client.pull_model.assert_not_called()

    # CA-12.2: Si el servicio de modelos no está disponible para un modelo dado, el sistema debe omitirlo (sin generar resultados ni intentar ejecutar el pipeline) y continuar con el resto.
    @patch("src.llm.evaluator.RAGPipeline")
    @patch("src.llm.evaluator.PromptBuilder")
    @patch("src.llm.evaluator.LLMClient")
    def test_evaluate_models_ollama_unavailable_skips_model(
        self, mock_llm_client_cls, mock_prompt_builder_cls, mock_rag_pipeline_cls
    ):
        from src.llm.evaluator import evaluate_models

        mock_client = MagicMock()
        mock_client.is_available.return_value = False
        mock_llm_client_cls.return_value = mock_client

        retriever = MagicMock()
        report = evaluate_models(
            retriever=retriever,
            models=["llama3:8b"],
            queries=[],
            n_runs=1,
        )

        assert report.results == []
        assert report.summary == {}
        assert report.selected_model == ""
        mock_rag_pipeline_cls.assert_not_called()

    # CA-12.3: Si un modelo indicado no está descargado localmente, el sistema debe descargarlo antes de evaluarlo.
    @patch("src.llm.evaluator.RAGPipeline")
    @patch("src.llm.evaluator.PromptBuilder")
    @patch("src.llm.evaluator.LLMClient")
    def test_evaluate_models_pulls_missing_model(
        self, mock_llm_client_cls, mock_prompt_builder_cls, mock_rag_pipeline_cls
    ):
        from src.llm.evaluator import evaluate_models

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.list_models.return_value = []  # el modelo no está descargado
        mock_llm_client_cls.return_value = mock_client

        mock_rag_pipeline_cls.return_value = self._make_mock_pipeline()

        retriever = MagicMock()
        evaluate_models(
            retriever=retriever,
            models=["llama3:8b"],
            queries=[],
            n_runs=1,
        )

        mock_client.pull_model.assert_called_once_with("llama3:8b")

    # CA-13.1: Si una consulta falla para un modelo, el sistema debe registrar el error asociado sin interrumpir la evaluación del resto de las consultas.
    # CA-13.2: El resumen por modelo debe reflejar la cantidad de errores ocurridos.
    @patch("src.llm.evaluator.RAGPipeline")
    @patch("src.llm.evaluator.PromptBuilder")
    @patch("src.llm.evaluator.LLMClient")
    def test_evaluate_models_records_error_on_exception(
        self, mock_llm_client_cls, mock_prompt_builder_cls, mock_rag_pipeline_cls
    ):
        from src.llm.evaluator import evaluate_models

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.list_models.return_value = ["llama3:8b"]
        mock_llm_client_cls.return_value = mock_client

        mock_rag_pipeline_cls.return_value = self._make_mock_pipeline(
            raise_on_query=True
        )

        retriever = MagicMock()
        queries = [{"id": "q1", "query": "test", "expected_keywords": ["x"]}]

        report = evaluate_models(
            retriever=retriever,
            models=["llama3:8b"],
            queries=queries,
            n_runs=1,
        )

        assert report.results[0].quality == 0.0
        assert "fallo de conexión" in report.results[0].error
        assert report.summary["llama3:8b"]["error_count"] == 1

    # CA-14.1: El sistema debe seleccionar como mejor modelo aquel con mayor score de calidad de respuesta, usando la latencia como criterio de desempate.
    # CA-14.2: El sistema debe generar una justificación textual de por qué se seleccionó ese modelo.
    @patch("src.llm.evaluator.RAGPipeline")
    @patch("src.llm.evaluator.PromptBuilder")
    @patch("src.llm.evaluator.LLMClient")
    def test_evaluate_models_selects_best_by_score_then_latency(
        self, mock_llm_client_cls, mock_rag_pipeline_cls_ignored, mock_rag_pipeline_cls
    ):
        # nota: RAGPipeline se instancia una vez por modelo; devolvemos
        # respuestas distintas según el modelo activo en cada iteración.
        from src.llm.evaluator import evaluate_models

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.list_models.return_value = ["modelo_bueno", "modelo_malo"]
        mock_llm_client_cls.return_value = mock_client

        good_pipeline = self._make_mock_pipeline(answer="presión psi fondo PM-104")
        bad_pipeline = self._make_mock_pipeline(answer="no tengo idea")
        mock_rag_pipeline_cls.side_effect = [good_pipeline, bad_pipeline]

        retriever = MagicMock()
        queries = [
            {
                "id": "q1",
                "query": "test",
                "expected_keywords": ["presión", "psi", "fondo", "PM-104"],
            }
        ]

        report = evaluate_models(
            retriever=retriever,
            models=["modelo_bueno", "modelo_malo"],
            queries=queries,
            n_runs=1,
        )

        assert report.selected_model == "modelo_bueno"
        assert "Calidad 100%" in report.selection_rationale
        assert "p50" in report.selection_rationale

    # CA-15.1: Si se indica una ruta de salida, el sistema debe guardar el reporte generado en esa ubicación.
    @patch("src.llm.evaluator.RAGPipeline")
    @patch("src.llm.evaluator.PromptBuilder")
    @patch("src.llm.evaluator.LLMClient")
    def test_evaluate_models_saves_report_when_output_path_given(
        self,
        mock_llm_client_cls,
        mock_prompt_builder_cls,
        mock_rag_pipeline_cls,
        tmp_path,
    ):
        from src.llm.evaluator import evaluate_models

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.list_models.return_value = ["llama3:8b"]
        mock_llm_client_cls.return_value = mock_client

        mock_rag_pipeline_cls.return_value = self._make_mock_pipeline()

        retriever = MagicMock()
        output_path = tmp_path / "report.json"

        evaluate_models(
            retriever=retriever,
            models=["llama3:8b"],
            queries=[],
            output_path=output_path,
            n_runs=1,
        )

        assert output_path.exists()

    # CA-22.1: Una consulta que falla debe contar como calidad cero para el modelo,
    # en lugar de quedar fuera del promedio.
    def test_successful_run_has_quality(self):
        from src.llm.evaluator import evaluate_models

        with patch("src.llm.evaluator.RAGPipeline") as pipeline_cls, patch(
            "src.llm.evaluator.PromptBuilder"
        ), patch("src.llm.evaluator.LLMClient") as client_cls:
            client = MagicMock()
            client.is_available.return_value = True
            client.list_models.return_value = ["m1"]
            client_cls.return_value = client
            pipeline_cls.return_value = self._make_mock_pipeline(answer="presión psi")

            report = evaluate_models(
                MagicMock(),
                ["m1"],
                queries=[
                    {
                        "id": "q1",
                        "query": "test",
                        "expected_keywords": ["presión", "psi"],
                        "n_runs": 1,
                    }
                ],
            )

        assert report.results[0].quality == 1.0


class TestRepetitions:
    # CA-27.3: El sistema debe poder repetir cada consulta varias veces por modelo,
    # y todas las repeticiones deben contarse en los resultados.

    def test_each_query_runs_n_times(self):
        from src.llm.evaluator import evaluate_models

        with patch("src.llm.evaluator.RAGPipeline") as pipeline_cls, patch(
            "src.llm.evaluator.PromptBuilder"
        ), patch("src.llm.evaluator.LLMClient") as client_cls:
            client = MagicMock()
            client.is_available.return_value = True
            client.list_models.return_value = ["m1"]
            client_cls.return_value = client
            pipeline = MagicMock()
            pipeline.query.return_value = SimpleNamespace(answer="presión")
            pipeline_cls.return_value = pipeline

            queries = [
                {"id": "a", "query": "x", "expected_keywords": ["presión"]},
                {"id": "b", "query": "y", "expected_keywords": ["presión"]},
            ]
            report = evaluate_models(MagicMock(), ["m1"], queries=queries, n_runs=3)

        assert {r.run for r in report.results} == {1, 2, 3}
        assert report.summary["m1"]["total_runs"] == 6


class TestNegativeQueries:
    # CA-25.2: Ante esas preguntas, el sistema debe considerar correcta la abstención
    # e incorrecta cualquier respuesta inventada, y el resumen debe reportar
    # la tasa de alucinación.
    def test_quality_of_negative_query(self):
        from src.llm.evaluator import _run_quality

        assert _run_quality(_result(should_abstain=True, abstained=True)) == 1.0
        assert _run_quality(_result(should_abstain=True, abstained=False)) == 0.0

    # CA-25.2: Ante esas preguntas, el sistema debe considerar correcta la abstención
    # e incorrecta cualquier respuesta inventada, y el resumen debe reportar
    # la tasa de alucinación.
    def test_summary_reports_hallucination_rate(self):
        from src.llm.evaluator import _summarize

        summary = _summarize(
            [
                _result(should_abstain=True, abstained=True),
                _result(should_abstain=True, abstained=False),
            ]
        )
        assert summary["hallucination_rate"] == 0.5
        assert summary["total_negative"] == 2

    # CA-25.2: Ante esas preguntas, el sistema debe considerar correcta la abstención
    # e incorrecta cualquier respuesta inventada, y el resumen debe reportar
    # la tasa de alucinación.
    def test_model_that_answers_negative_query_hallucinates(self, mocks):
        from src.llm.evaluator import evaluate_models

        mocks.pipeline_cls.return_value = _pipeline_by_query(
            {**ANSWERS, "¿Qué pasó en ZZ-999?": "Falló por fatiga."}
        )
        report = evaluate_models(MagicMock(), ["m1"], queries=QUERIES, n_runs=1)

        assert report.summary["m1"]["hallucination_rate"] == 1.0

    # CA-25.3: Ante las preguntas que sí tienen respuesta,
    # el resumen debe reportar la tasa de abstención incorrecta,
    # calculada solo sobre las consultas ejecutadas sin error.
    def test_false_abstention_ignores_failed_runs(self):
        from src.llm.evaluator import _summarize

        results = [_result(abstained=True), _result(abstained=False, error="boom")]
        assert _summarize(results)["false_abstention_rate"] == 1.0

    # CA-25.4: Si el conjunto evaluado no incluye preguntas sin respuesta,
    # el resumen en consola debe indicarlo en lugar de mostrar una tasa.
    def test_print_summary_without_negatives_shows_nd(self, mocks, capsys):
        from src.llm.evaluator import evaluate_models

        evaluate_models(MagicMock(), ["m1"], queries=QUERIES[:1], n_runs=1)

        assert "n/d" in capsys.readouterr().out


class TestExecutionRobustness:
    # CA-29.1: Si un modelo ya está descargado,
    # el sistema no debe volver a descargarlo aunque se lo nombre sin indicar la versión.
    @pytest.mark.parametrize(
        "model,available,expected",
        [
            ("llama3.1", ["llama3.1:latest"], True),
            ("llama3.1", ["llama3.1:8b"], False),
            ("llama3.1:8b", ["llama3.1:latest"], False),
            ("qwen3:8b", ["qwen3:8b"], True),
            ("qwen3:8b", [], False),
        ],
    )
    def test_model_available_tolerates_latest_tag(self, model, available, expected):
        from src.llm.evaluator import _model_available

        assert _model_available(model, available) is expected

    # CA-29.1: Si un modelo ya está descargado,
    # el sistema no debe volver a descargarlo aunque se lo nombre sin indicar la versión.
    def test_does_not_pull_when_installed_as_latest(self, mocks):
        from src.llm.evaluator import evaluate_models

        mocks.client.list_models.return_value = ["m1:latest"]
        evaluate_models(MagicMock(), ["m1"], queries=QUERIES, n_runs=1)
        mocks.client.pull_model.assert_not_called()

    # CA-29.2: Si un modelo no está disponible, el sistema debe omitirlo
    # y continuar con los demás.
    def test_unavailable_model_is_skipped_and_others_continue(self, mocks):
        from src.llm.evaluator import evaluate_models

        def _factory(config):
            client = _mock_client(["m1", "m2"])
            client.is_available.return_value = config.model != "m2"
            return client

        mocks.client_cls.side_effect = _factory
        report = evaluate_models(MagicMock(), ["m1", "m2"], queries=QUERIES, n_runs=1)

        assert "m2" not in report.summary
        assert "m1" in report.summary

    # CA-29.3: El sistema debe guardar el reporte completo en un archivo,
    # creando las carpetas necesarias si no existen.
    def test_report_is_saved_as_valid_json(self, mocks, tmp_path):
        from src.llm.evaluator import evaluate_models

        output = tmp_path / "sub" / "eval.json"
        report = evaluate_models(
            MagicMock(), ["m1"], queries=QUERIES, n_runs=1, output_path=output
        )

        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["selected_model"] == report.selected_model
        assert len(data["results"]) == 2


class TestJudge:
    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_judge_scores_are_recorded(self, mocks):
        from src.llm.evaluator import evaluate_models

        report = evaluate_models(
            MagicMock(), ["m1"], queries=QUERIES, n_runs=1, judge_model="judge"
        )
        r = report.results[0]
        assert r.judge_correctness == 5.0
        assert r.judge_completeness == 4.0
        assert r.judge_faithfulness == 5.0

    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_judge_abstention_verdict_is_used(self, mocks):
        from src.llm.evaluator import evaluate_models

        mocks.client.generate.return_value = '{"correctness": 1, "completeness": 1, "faithfulness": 5, "abstained": true}'
        report = evaluate_models(
            MagicMock(), ["m1"], queries=QUERIES, n_runs=1, judge_model="judge"
        )
        assert report.results[0].abstained is True

    # CA-24.2: Si el juez falla o devuelve una evaluación inválida,
    # la evaluación debe continuar sin esos puntajes.
    def test_invalid_judge_output_does_not_break_evaluation(self, mocks):
        from src.llm.evaluator import evaluate_models

        mocks.client.generate.return_value = "esto no es json"
        report = evaluate_models(
            MagicMock(), ["m1"], queries=QUERIES, n_runs=1, judge_model="judge"
        )
        assert report.results[0].ok
        assert report.results[0].judge_correctness is None
        assert report.results[1].abstained is True

    # CA-24.2: Si el juez falla o devuelve una evaluación inválida,
    # la evaluación debe continuar sin esos puntajes.
    def test_judge_exception_does_not_break_evaluation(self, mocks):
        from src.llm.evaluator import evaluate_models

        mocks.client.generate.side_effect = RuntimeError("juez caído")
        report = evaluate_models(
            MagicMock(), ["m1"], queries=QUERIES, n_runs=1, judge_model="judge"
        )
        assert report.results[0].ok
        assert report.results[0].judge_correctness is None

    # CA-24.3: Si no se indica un modelo juez, el sistema debe evaluar igual,
    # detectando la abstención por el texto de la respuesta.
    def test_without_judge_uses_text_patterns(self, mocks):
        from src.llm.evaluator import evaluate_models

        report = evaluate_models(MagicMock(), ["m1"], queries=QUERIES, n_runs=1)

        assert report.results[0].judge_correctness is None
        assert report.results[1].abstained is True
        mocks.client.generate.assert_not_called()

    # CA-24.4: Si el modelo juez es uno de los modelos evaluados,
    # el sistema debe advertirlo, porque se estaría juzgando a sí mismo.
    def test_warns_when_judge_is_also_an_evaluated_model(self, mocks, caplog):
        from src.llm.evaluator import evaluate_models

        with caplog.at_level("WARNING"):
            evaluate_models(
                MagicMock(), ["m1"], queries=QUERIES, n_runs=1, judge_model="m1"
            )
        assert "se juzgaría a sí mismo" in caplog.text

    # CA-24.4: Si el modelo juez es uno de los modelos evaluados,
    # el sistema debe advertirlo, porque se estaría juzgando a sí mismo.
    def test_no_warning_when_judge_is_a_different_model(self, mocks, caplog):
        from src.llm.evaluator import evaluate_models

        with caplog.at_level("WARNING"):
            evaluate_models(
                MagicMock(), ["m1"], queries=QUERIES, n_runs=1, judge_model="judge"
            )
        assert "se juzgaría a sí mismo" not in caplog.text


class TestRetrieverSeparation:
    # CA-26.1: El sistema debe guardar, para cada consulta,
    # la información que el buscador entregó al modelo.
    def test_result_stores_retrieved_context(self, mocks):
        from src.llm.evaluator import evaluate_models

        report = evaluate_models(MagicMock(), ["m1"], queries=QUERIES, n_runs=1)
        assert "LCM" in report.results[0].context

    # CA-26.2: El sistema debe indicar si esa información
    # contenía los datos necesarios para responder.
    def test_context_recall_and_flag(self, mocks):
        from src.llm.evaluator import evaluate_models

        report = evaluate_models(MagicMock(), ["m1"], queries=QUERIES, n_runs=1)
        r = report.results[0]
        assert r.context_recall == 1.0
        assert r.context_ok is True

    # CA-26.2: El sistema debe indicar si esa información
    # contenía los datos necesarios para responder.
    def test_context_without_the_answer_is_flagged(self, mocks):
        from src.llm.evaluator import evaluate_models

        pipeline = MagicMock()
        pipeline.query.return_value = _rag_resp(
            "respuesta", context="texto sin relación alguna"
        )
        mocks.pipeline_cls.return_value = pipeline

        report = evaluate_models(MagicMock(), ["m1"], queries=QUERIES[:1], n_runs=1)
        assert report.results[0].context_ok is False


class TestLatencyMeasurement:
    # CA-27.1: Antes de medir cada modelo, el sistema debe hacer
    # una consulta de calentamiento que no cuente en los resultados,
    # para que la carga inicial del modelo no distorsione los tiempos.
    def test_warmup_is_not_counted_in_results(self, mocks):
        from src.llm.evaluator import evaluate_models

        pipeline = _pipeline_by_query(ANSWERS)
        mocks.pipeline_cls.return_value = pipeline

        report = evaluate_models(MagicMock(), ["m1"], queries=QUERIES, n_runs=2)

        assert len(report.results) == 4
        assert pipeline.query.call_count == 5  # 4 corridas + 1 calentamiento

    # CA-27.4: La evaluación del modelo juez debe realizarse una vez terminadas
    # las consultas a todos los modelos, para que no distorsione
    # los tiempos de respuesta medidos ni obligue a cargar
    # y descargar modelos de forma repetida.
    def test_judge_runs_after_all_models_finish(self, mocks):
        from src.llm.evaluator import evaluate_models

        calls = []

        def _query(text):
            calls.append("query")
            return _rag_resp("respuesta")

        def _judge(prompt):
            calls.append("judge")
            return '{"correctness": 5, "completeness": 5, "faithfulness": 5, "abstained": false}'

        pipeline = MagicMock()
        pipeline.query.side_effect = _query
        mocks.pipeline_cls.return_value = pipeline
        mocks.client.list_models.return_value = ["m1", "m2", "judge"]
        mocks.client.generate.side_effect = _judge

        evaluate_models(
            MagicMock(), ["m1", "m2"], queries=QUERIES, n_runs=1, judge_model="judge"
        )

        first_judge = calls.index("judge")
        assert "query" not in calls[first_judge:]
        assert calls.count("judge") == 4  # 2 modelos x 2 consultas
