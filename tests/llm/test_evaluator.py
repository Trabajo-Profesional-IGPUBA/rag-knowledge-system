from __future__ import annotations


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
