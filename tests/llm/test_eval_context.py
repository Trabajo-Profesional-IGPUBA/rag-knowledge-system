"""
Cubre "Mejorar la selección de modelos LLM en la evaluación RAG: scoring más robusto, medición de latencia confiable y elección que combina calidad y tiempo de respuesta"
  - Información recuperada por el buscador-> CA-26.1 y CA-26.4
"""

from types import SimpleNamespace

import pytest

from tests.llm.helpers import (
    _rag_resp,
)


class TestEvalContext:
    # CA-26.4: Si no se puede obtener el texto de la información recuperada,
    # la consulta se registra como error.
    def test_context_without_text_is_an_error(self):
        from src.llm.eval_context import _extract_context

        resp = SimpleNamespace(
            retrieval=SimpleNamespace(chunks=[]), prompt=SimpleNamespace(num_chunks=0)
        )
        with pytest.raises(RuntimeError):
            _extract_context(resp)

    # CA-26.1: El sistema debe guardar, para cada consulta,
    # la información que el buscador entregó al modelo.
    @pytest.mark.parametrize(
        "resp,expected",
        [
            (_rag_resp("r", "a"), "a"),
            (
                SimpleNamespace(
                    retrieval=SimpleNamespace(chunks=[{"text": "a"}, {"text": "b"}]),
                    prompt=SimpleNamespace(num_chunks=1),
                ),
                "a",
            ),
            (
                SimpleNamespace(
                    retrieval=SimpleNamespace(chunks=[{"text": "a"}, {"text": "b"}]),
                    prompt=SimpleNamespace(num_chunks=2),
                ),
                "a\n---\nb",
            ),
        ],
    )
    def test_extract_context_shapes(self, resp, expected):
        from src.llm.eval_context import _extract_context

        assert _extract_context(resp) == expected
