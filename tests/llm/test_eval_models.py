"""
Cubre "ÉPICA: Cliente LLM sobre Ollama":
  - Definición de resultados y reportes -> CA-10.1 Y  CA-10.4
"""


def _result(**overrides):
    from src.llm.evaluator import ModelEvalResult

    base = {
        "model": "m",
        "query_id": "q1",
        "query": "test",
        "response": "respuesta",
        "elapsed_sec": 1.0,
        "response_length": 9,
        "keyword_hits": 0,
        "keyword_total": 0,
        "keyword_score": 0.0,
    }
    base.update(overrides)
    return ModelEvalResult(**base)


class TestEvaluator:
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
        from src.llm.evaluator import EvaluationReport, _summarize

        report = EvaluationReport(models_evaluated=["llama3:8b"])
        report.summary["llama3:8b"] = _summarize(
            [
                _result(
                    model="llama3:8b",
                    elapsed_sec=2.5,
                    response_length=100,
                    keyword_hits=3,
                    keyword_total=4,
                    keyword_score=0.75,
                )
            ]
        )
        report.selected_model = "llama3:8b"
        report.selection_rationale = "Mejor balance calidad/latencia."

        report.print_summary()

        out = capsys.readouterr().out
        assert "llama3:8b" in out
        assert "75%" in out
        assert "2.50s" in out
        assert "100 chars" in out
        assert "Mejor balance calidad/latencia." in out
