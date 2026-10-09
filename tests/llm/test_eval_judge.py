"""
Cubre "Mejorar la selección de modelos LLM en la evaluación RAG: scoring más robusto, medición de latencia confiable y elección que combina calidad y tiempo de respuesta"
  - Calidad de cada respuesta-> CA-23.1
  - Evaluación por un modelo juez-> CA-24.1 a CA-24.2
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.llm.helpers import (
    _judge_client,
    _result,
)


class TestJudgeGenerate:
    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_generate_reads_text_from_response_object(self):
        from src.llm.eval_judge import _generate

        client = MagicMock()
        client.generate.return_value = SimpleNamespace(text="hola")
        assert _generate(client, "prompt") == "hola"

    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_generate_accepts_plain_string(self):
        from src.llm.eval_judge import _generate

        client = MagicMock()
        client.generate.return_value = "hola"
        assert _generate(client, "prompt") == "hola"


class TestJudgeVerdict:
    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_judge_returns_scores(self):
        from src.llm.eval_judge import _judge

        client = MagicMock()
        client.generate.return_value = '{"correctness": 5, "completeness": 4, "faithfulness": 5, "abstained": false}'
        verdict = _judge(client, "q", "resp", "ctx", "ref", False)
        assert verdict["correctness"] == 5
        assert verdict["abstained"] is False

    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_judge_extracts_json_surrounded_by_text(self):
        from src.llm.eval_judge import _judge

        client = MagicMock()
        client.generate.return_value = 'Acá va: {"correctness": 3} listo'
        assert _judge(client, "q", "resp", "ctx", "ref", False) == {"correctness": 3}

    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_prompt_for_negative_query_expects_abstention(self):
        from src.llm.eval_judge import _judge

        client = MagicMock()
        client.generate.return_value = "{}"
        _judge(client, "q", "resp", "ctx", None, True)
        assert "NO tiene respuesta" in client.generate.call_args.args[0]

    # CA-24.2: Si el juez falla o devuelve una evaluación inválida,
    # la evaluación debe continuar sin esos puntajes.
    def test_invalid_output_returns_none(self):
        from src.llm.eval_judge import _judge

        client = MagicMock()
        client.generate.return_value = "esto no es json"
        assert _judge(client, "q", "resp", "ctx", "ref", False) is None

    # CA-24.2: Si el juez falla o devuelve una evaluación inválida,
    # la evaluación debe continuar sin esos puntajes.
    def test_broken_json_returns_none(self):
        from src.llm.eval_judge import _judge

        client = MagicMock()
        client.generate.return_value = "{correctness: cinco}"
        assert _judge(client, "q", "resp", "ctx", "ref", False) is None

    # CA-24.2: Si el juez falla o devuelve una evaluación inválida,
    # la evaluación debe continuar sin esos puntajes.
    def test_exception_returns_none(self):
        from src.llm.eval_judge import _judge

        client = MagicMock()
        client.generate.side_effect = RuntimeError("juez caído")
        assert _judge(client, "q", "resp", "ctx", "ref", False) is None


class TestApplyJudge:
    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_scores_are_recorded(self):
        from src.llm.eval_judge import _apply_judge

        r = _result()
        client = _judge_client(
            '{"correctness": 5, "completeness": 4, "faithfulness": 5, "abstained": false}'
        )
        _apply_judge(r, {"reference_answer": "ref"}, client)

        assert r.judge_correctness == 5.0
        assert r.judge_completeness == 4.0
        assert r.judge_faithfulness == 5.0

    # CA-24.1: Si se indica un modelo juez, el sistema debe puntuar cada respuesta
    # en corrección, completitud y fidelidad al contexto,
    # e indicar si el modelo se abstuvo.
    def test_abstention_verdict_overrides_pattern_detection(self):
        from src.llm.eval_judge import _apply_judge

        r = _result(abstained=False)
        client = _judge_client(
            '{"correctness": 1, "completeness": 1, "faithfulness": 5, "abstained": true}'
        )
        _apply_judge(r, {}, client)

        assert r.abstained is True

    # CA-23.1: El sistema debe calcular la calidad de cada respuesta combinando
    # las medidas disponibles (evaluación del juez, palabras clave, cifras
    # y similitud semántica), usando solo las que se pudieron obtener.
    def test_quality_is_recomputed_with_judge_score(self):
        from src.llm.eval_judge import _apply_judge

        r = _result()
        _apply_judge(r, {}, _judge_client('{"correctness": 5}'))

        assert r.quality == pytest.approx(1.0)

    # CA-24.2: Si el juez falla o devuelve una evaluación inválida,
    # la evaluación debe continuar sin esos puntajes.
    def test_invalid_verdict_leaves_result_untouched(self):
        from src.llm.eval_judge import _apply_judge

        r = _result(abstained=True, quality=0.3)
        _apply_judge(r, {}, _judge_client("esto no es json"))

        assert r.judge_correctness is None
        assert r.abstained is True
        assert r.quality == 0.3

    # CA-24.2: Si el juez falla o devuelve una evaluación inválida,
    # la evaluación debe continuar sin esos puntajes.
    def test_non_numeric_score_is_ignored(self):
        from src.llm.eval_judge import _apply_judge

        r = _result()
        _apply_judge(r, {}, _judge_client('{"correctness": "alto"}'))

        assert r.judge_correctness is None
