from unittest.mock import MagicMock, patch

"""
Cubre "ÉPICA: Cliente LLM sobre Ollama":
  - Definición de resultados y reportes -> CA-10.1 Y  CA-10.4
  - Medición de calidad de respuesta -> CA-11.1 a CA-11.3
  - Ejecución de la evaluación -> CA-12.1 a CA-12.4
  - Manejo de errores durante la evaluación -> CA-13.1 a CA-13.2
  - Selección del modelo -> CA-14.1 a CA-14.2
  - Persistencia de resultados -> CA-15.1 

Cubre "Mejorar precisión del scoring en la evaluación de LLMs: el matching exacto de keywords 
subestimaba la calidad real de las respuestas":
  - Normalización de keywords -> CA-16.1
  - Detección de abstención -> CA-17.1 a 17.2
  - Similitud semántica-> CA-18.1 a 18.5
  - Selección del modelo con criterio combinado -> CA-19.1 a 19.2

"""


class TestEvaluator:
    # CA-11.1: El sistema debe calcular qué proporción de palabras clave esperadas aparece en una respuesta generada.
    def test_keyword_scoring(self):
        from src.llm.evaluator import _score_keywords

        response = "La presión de fondo del pozo PM-104 es 3500 psi"
        keywords = ["presión", "psi", "fondo", "PM-104"]
        hits, score = _score_keywords(response, keywords)
        assert hits == 4
        assert score == 1.0

    # CA-11.2: El sistema debe calcular correctamente el score cuando solo una parte de las palabras clave está presente.
    def test_keyword_scoring_partial(self):
        from src.llm.evaluator import _score_keywords

        response = "La presión es alta"
        keywords = ["presión", "psi", "fondo"]
        hits, score = _score_keywords(response, keywords)
        assert hits == 1
        assert round(score, 4) == round(1 / 3, 4)

    # CA-11.3: Si no hay palabras clave esperadas para una consulta, el sistema no debe fallar y debe devolver un score de cero.
    def test_keyword_scoring_empty_keywords(self):
        from src.llm.evaluator import _score_keywords

        hits, score = _score_keywords("cualquier texto", [])
        assert hits == 0
        assert score == 0.0

    # CA-10.1: El sistema debe registrar si una evaluación puntual fue exitosa o no, según si tiene un error asociado.
    def test_evaluation_report_summary(self):
        from src.llm.evaluator import EvaluationReport, ModelEvalResult

        report = EvaluationReport(models_evaluated=["llama3:8b"])
        report.results = [
            ModelEvalResult(
                model="llama3:8b",
                query_id="q1",
                query="test",
                response="respuesta",
                elapsed_sec=2.5,
                response_length=100,
                keyword_hits=3,
                keyword_total=4,
                keyword_score=0.75,
            )
        ]
        report.summary["llama3:8b"] = {
            "avg_elapsed_sec": 2.5,
            "avg_keyword_score": 0.75,
            "avg_response_length": 100.0,
            "error_count": 0,
            "total_queries": 1,
        }
        report.selected_model = "llama3:8b"
        assert report.summary["llama3:8b"]["avg_keyword_score"] == 0.75

    # CA-10.1: El sistema debe registrar si una evaluación puntual fue exitosa o no, según si tiene un error asociado.
    def test_model_eval_result_ok_true_without_error(self):
        from src.llm.evaluator import ModelEvalResult

        result = ModelEvalResult(
            model="llama3:8b",
            query_id="q1",
            query="test",
            response="respuesta",
            elapsed_sec=1.0,
            response_length=10,
            keyword_hits=1,
            keyword_total=1,
            keyword_score=1.0,
        )
        assert result.ok is True

    # CA-10.1: El sistema debe registrar si una evaluación puntual fue exitosa o no, según si tiene un error asociado.
    def test_model_eval_result_ok_false_with_error(self):
        from src.llm.evaluator import ModelEvalResult

        result = ModelEvalResult(
            model="llama3:8b",
            query_id="q1",
            query="test",
            response="",
            elapsed_sec=0.0,
            response_length=0,
            keyword_hits=0,
            keyword_total=1,
            keyword_score=0.0,
            error="timeout",
        )
        assert result.ok is False

    # CA-10.2: El sistema debe poder convertir el reporte completo (resultados, resumen, modelo seleccionado) a un formato exportable.
    def test_evaluation_report_to_dict(self):
        from src.llm.evaluator import EvaluationReport, ModelEvalResult

        report = EvaluationReport(models_evaluated=["llama3:8b"])
        report.results = [
            ModelEvalResult(
                model="llama3:8b",
                query_id="q1",
                query="test",
                response="respuesta",
                elapsed_sec=1.0,
                response_length=9,
                keyword_hits=1,
                keyword_total=1,
                keyword_score=1.0,
            )
        ]
        d = report.to_dict()
        assert d["models_evaluated"] == ["llama3:8b"]
        assert d["results"][0]["model"] == "llama3:8b"
        assert "evaluated_at" in d

    # CA-10.3: El sistema debe poder guardar el reporte en un archivo, creando las carpetas necesarias si no existen.
    def test_evaluation_report_save(self, tmp_path):
        from src.llm.evaluator import EvaluationReport

        report = EvaluationReport(models_evaluated=["llama3:8b"])
        report.selected_model = "llama3:8b"
        output_path = tmp_path / "subdir" / "report.json"

        report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "llama3:8b" in content

    # CA-10.4: El sistema debe poder mostrar en consola un resumen legible por modelo, incluyendo latencia, score de calidad, longitud de respuesta, errores y el modelo seleccionado con su justificación.
    def test_evaluation_report_print_summary(self, capsys):
        from src.llm.evaluator import EvaluationReport

        report = EvaluationReport(models_evaluated=["llama3:8b"])
        report.summary["llama3:8b"] = {
            "avg_elapsed_sec": 2.5,
            "avg_keyword_score": 0.75,
            "avg_response_length": 100.0,
            "error_count": 0,
        }
        report.selected_model = "llama3:8b"
        report.selection_rationale = "Mejor balance calidad/latencia."

        report.print_summary()

        captured = capsys.readouterr()
        assert "llama3:8b" in captured.out
        assert "75%" in captured.out
        assert "Mejor balance calidad/latencia." in captured.out


class TestEvaluateModels:
    def _make_mock_pipeline(
        self, answer="La presión es 3500 psi", raise_on_query=False
    ):
        pipeline = MagicMock()
        if raise_on_query:
            pipeline.query.side_effect = RuntimeError("fallo de conexión")
        else:
            rag_resp = MagicMock()
            rag_resp.answer = answer
            pipeline.query.return_value = rag_resp
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
            retriever=retriever, models=["llama3:8b"], queries=queries
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
        report = evaluate_models(retriever=retriever, models=["llama3:8b"], queries=[])

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
        evaluate_models(retriever=retriever, models=["llama3:8b"], queries=[])

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
            retriever=retriever, models=["llama3:8b"], queries=queries
        )

        assert report.results[0].ok is False
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
            retriever=retriever, models=["modelo_bueno", "modelo_malo"], queries=queries
        )

        assert report.selected_model == "modelo_bueno"
        assert "Mayor score de relevancia" in report.selection_rationale

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
        )

        assert output_path.exists()


class TestNormalization:
    # CA-16.1: El sistema debe normalizar acentos y mayúsculas al comparar palabras
    # clave esperadas contra la respuesta generada.
    def test_normalize_removes_accents_and_lowercases(self):
        from src.llm.evaluator import _normalize

        assert _normalize("Pérdida de Circulación") == "perdida de circulacion"

    # CA-16.1: El sistema debe normalizar acentos y mayúsculas al comparar palabras
    # clave esperadas contra la respuesta generada, para que variaciones ortográficas
    # no reduzcan el score injustamente.
    def test_score_keywords_matches_despite_accent_mismatch(self):
        from src.llm.evaluator import _score_keywords

        response = "Hubo perdida de circulacion en el pozo"
        keywords = ["pérdida de circulación"]
        hits, score = _score_keywords(response, keywords)
        assert hits == 1
        assert score == 1.0


class TestNoInfoDetection:
    # CA-17.1: El sistema debe distinguir cuándo una respuesta corresponde a una abstención explícita del modelo
    # (ej. "no encontré información"), en lugar de tratarla igual que una respuesta con contenido incorrecto.

    def test_detects_no_info_response(self):
        from src.llm.evaluator import _is_no_info_response

        assert _is_no_info_response(
            "No encontré información sobre esto en los documentos disponibles."
        )

    # CA-17.1: El sistema debe distinguir cuándo una respuesta corresponde a una abstención explícita del modelo
    # (ej. "no encontré información"), en lugar de tratarla igual que una respuesta con contenido incorrecto.
    def test_does_not_flag_normal_response(self):
        from src.llm.evaluator import _is_no_info_response

        assert not _is_no_info_response(
            "El pozo PM-104 tuvo pérdida de circulación en Quintuco."
        )


class TestSummaryNoInfoRate:
    # CA-17.2: El resumen por modelo debe reportar la tasa de abstención, calculada
    # solo sobre las consultas ejecutadas sin error.
    def test_summary_includes_no_info_rate(self):
        from src.llm.evaluator import EvaluationReport, ModelEvalResult

        report = EvaluationReport(models_evaluated=["modelo_x"])
        report.results = [
            ModelEvalResult(
                model="modelo_x",
                query_id="q1",
                query="test",
                response="No encontré información sobre esto.",
                elapsed_sec=1.0,
                response_length=30,
                keyword_hits=0,
                keyword_total=3,
                keyword_score=0.0,
                is_no_info_response=True,
            ),
            ModelEvalResult(
                model="modelo_x",
                query_id="q2",
                query="test2",
                response="Respuesta con contenido real",
                elapsed_sec=2.0,
                response_length=30,
                keyword_hits=2,
                keyword_total=3,
                keyword_score=0.67,
                is_no_info_response=False,
            ),
        ]
        report.summary["modelo_x"] = {
            "avg_elapsed_sec": 1.5,
            "avg_keyword_score": 0.335,
            "no_info_rate": 0.5,
            "avg_response_length": 30.0,
            "error_count": 0,
            "total_queries": 2,
        }
        assert report.summary["modelo_x"]["no_info_rate"] == 0.5


class TestSemanticSimilarity:
    # CA-18.2: Si no hay respuesta de referencia, el cálculo de similitud semántica
    # debe devolver None sin fallar.
    def test_returns_none_without_reference(self):
        from src.llm.evaluator import _semantic_similarity

        result = _semantic_similarity("cualquier respuesta", None, MagicMock())
        assert result is None

    # CA-18.4: Si no se provee un embedder, el cálculo de similitud semántica debe
    # devolver None sin fallar.
    def test_returns_none_without_embedder(self):
        from src.llm.evaluator import _semantic_similarity

        result = _semantic_similarity("respuesta", "referencia", None)
        assert result is None

    # CA-18.1: El sistema debe poder calcular la similitud semántica entre la
    # respuesta generada y una respuesta de referencia, cuando esta última esté
    # definida para la consulta.
    def test_identical_texts_have_similarity_close_to_one(self):
        from src.llm.evaluator import _semantic_similarity

        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]

        result = _semantic_similarity("mismo texto", "mismo texto", embedder)
        assert result == 1.0

    # CA-18.5: El cálculo de similitud semántica (coseno) no debe fallar ante un
    # vector nulo, debe devolver 0.0 en ese caso.
    def test_cosine_similarity_zero_vector_returns_zero(self):
        from src.llm.evaluator import _cosine_similarity

        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestSummarySemanticSimilarity:
    # CA-18.3: El resumen por modelo debe reportar el promedio de similitud
    # semántica sobre las consultas donde pudo calcularse.
    def test_summary_averages_only_available_similarity_values(self):
        from src.llm.evaluator import EvaluationReport, ModelEvalResult

        report = EvaluationReport(models_evaluated=["modelo_x"])
        report.results = [
            ModelEvalResult(
                model="modelo_x",
                query_id="q1",
                query="test",
                response="resp 1",
                elapsed_sec=1.0,
                response_length=10,
                keyword_hits=1,
                keyword_total=2,
                keyword_score=0.5,
                semantic_similarity=0.9,
            ),
            ModelEvalResult(
                model="modelo_x",
                query_id="q2",
                query="test2",
                response="resp 2",
                elapsed_sec=1.0,
                response_length=10,
                keyword_hits=1,
                keyword_total=2,
                keyword_score=0.5,
                semantic_similarity=None,  # sin reference_answer para esta query
            ),
        ]
        report.summary["modelo_x"] = {
            "avg_elapsed_sec": 1.0,
            "avg_keyword_score": 0.5,
            "avg_semantic_similarity": 0.9,  # promedio solo sobre el valor disponible
            "avg_response_length": 10.0,
            "error_count": 0,
            "total_queries": 2,
        }
        assert report.summary["modelo_x"]["avg_semantic_similarity"] == 0.9


class TestSelectionCriteria:
    # CA-19.1: El sistema debe seleccionar el mejor modelo combinando el score de
    # keywords y la similitud semántica como medida de calidad, usando la latencia
    # como criterio de desempate.
    # CA-19.2: La justificación de selección debe reportar por separado el score
    # de keywords, la similitud semántica y la tasa de abstención del modelo elegido.
    def test_selects_model_with_best_combined_quality(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "modelo_a": {
                "avg_elapsed_sec": 10.0,
                "avg_keyword_score": 0.3,
                "avg_semantic_similarity": 0.3,
                "no_info_rate": 0.5,
            },
            "modelo_b": {
                "avg_elapsed_sec": 20.0,
                "avg_keyword_score": 0.8,
                "avg_semantic_similarity": 0.8,
                "no_info_rate": 0.0,
            },
        }

        selected, rationale = _select_best_model(summary)

        assert selected == "modelo_b"
        assert "80%" in rationale
        assert "Tasa de abstención: 0%" in rationale

    # CA-19.1: El sistema debe seleccionar el mejor modelo combinando el score de
    # keywords y la similitud semántica como medida de calidad, usando la latencia
    # como criterio de desempate.
    def test_selects_lower_latency_on_quality_tie(self):
        from src.llm.evaluator import _select_best_model

        summary = {
            "modelo_rapido": {
                "avg_elapsed_sec": 5.0,
                "avg_keyword_score": 0.5,
                "avg_semantic_similarity": 0.5,
                "no_info_rate": 0.0,
            },
            "modelo_lento": {
                "avg_elapsed_sec": 50.0,
                "avg_keyword_score": 0.5,
                "avg_semantic_similarity": 0.5,
                "no_info_rate": 0.0,
            },
        }

        selected, _ = _select_best_model(summary)

        assert selected == "modelo_rapido"
