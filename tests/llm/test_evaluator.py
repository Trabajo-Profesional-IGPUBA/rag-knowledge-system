from unittest.mock import MagicMock, patch


class TestEvaluator:
    def test_keyword_scoring(self):
        from src.llm.evaluator import _score_keywords

        response = "La presión de fondo del pozo PM-104 es 3500 psi"
        keywords = ["presión", "psi", "fondo", "PM-104"]
        hits, score = _score_keywords(response, keywords)
        assert hits == 4
        assert score == 1.0

    def test_keyword_scoring_partial(self):
        from src.llm.evaluator import _score_keywords

        response = "La presión es alta"
        keywords = ["presión", "psi", "fondo"]
        hits, score = _score_keywords(response, keywords)
        assert hits == 1
        assert round(score, 4) == round(1 / 3, 4)

    def test_keyword_scoring_empty_keywords(self):
        from src.llm.evaluator import _score_keywords

        hits, score = _score_keywords("cualquier texto", [])
        assert hits == 0
        assert score == 0.0

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

    def test_evaluation_report_save(self, tmp_path):
        from src.llm.evaluator import EvaluationReport

        report = EvaluationReport(models_evaluated=["llama3:8b"])
        report.selected_model = "llama3:8b"
        output_path = tmp_path / "subdir" / "report.json"

        report.save(output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "llama3:8b" in content

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
