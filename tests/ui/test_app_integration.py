"""Tests de integración: app.py + RAGPipeline + PromptBuilder reales.

Solo se mockea la infraestructura pesada (Embedder, VectorStore, Retriever,
LLMClient). El objetivo es verificar que las piezas reales se conectan bien
entre sí, cosa que los tests unitarios de RAGPipeline y los tests de UI con
RAGPipeline mockeado no pueden garantizar por separado.

"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from src.retrieval.retriever import RetrievalResult

APP_PATH = str(Path(__file__).resolve().parents[2] / "app.py")

HISTORY_HEADER = "HISTORIAL DE CONVERSACIÓN"


class TestChatAppIntegration:
    def setup_method(self):
        import streamlit as st

        st.cache_resource.clear()

    def _make_fake_retriever(self, top_k: int = 5):
        """Retriever falso que siempre devuelve un chunk con score alto,
        para no depender de embeddings/vectorstore reales."""
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = RetrievalResult(
            query="",
            top_k=top_k,
            chunks=[
                {
                    "chunk_id": "c1",
                    "score": 0.9,
                    "text": "La presión del pozo CH-88 es 3500 psi.",
                    "metadata": {"filename": "pozo_ch88.pdf", "doc_type": "informe"},
                }
            ],
        )
        return fake_retriever

    def _make_fake_llm(self, responses: list[list[str]]):
        """LLMClient falso: cada llamada a generate_stream consume la
        siguiente lista de tokens de `responses`, en orden, y registra
        el prompt final recibido en cada llamada."""
        fake_llm = MagicMock()
        fake_llm.is_available.return_value = True
        fake_llm.config.model = "llama3:8b"

        call_log: list[str] = []
        responses_iter = iter(responses)

        def _generate_stream(prompt):
            call_log.append(prompt)
            return iter(next(responses_iter))

        fake_llm.generate_stream.side_effect = _generate_stream
        fake_llm._call_log = call_log
        return fake_llm

    def _patch_real_pipeline(self, fake_retriever, fake_llm):
        """Parchea solo la infraestructura pesada. RAGPipeline y PromptBuilder
        quedan reales, no se mockean."""
        return {
            "embedder": patch(
                "src.embeddings.embedder.Embedder", return_value=MagicMock()
            ),
            "vectorstore": patch(
                "src.retrieval.vectorstore.VectorStore", return_value=MagicMock()
            ),
            "retriever": patch(
                "src.retrieval.retriever.Retriever", return_value=fake_retriever
            ),
            "llm": patch("src.llm.client.LLMClient", return_value=fake_llm),
        }

    def _start_all(self, named_patches: dict) -> dict:
        return {name: p.start() for name, p in named_patches.items()}

    def _stop_all(self, named_patches: dict) -> None:
        for p in named_patches.values():
            p.stop()

    def test_first_question_has_no_history_in_prompt(self):
        """Una pregunta nueva, sin conversación previa, se responde
        con normalidad y el prompt real (armado por RAGPipeline + PromptBuilder
        reales) no incluye historial de ningún tipo."""
        fake_retriever = self._make_fake_retriever()
        fake_llm = self._make_fake_llm(responses=[["La ", "presión ", "es 3500 psi"]])
        named_patches = self._patch_real_pipeline(fake_retriever, fake_llm)
        self._start_all(named_patches)
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()
            at.chat_input[0].set_value("¿Cuál es la presión del pozo CH-88?").run()

            assert (
                at.session_state["messages"][-1]["content"] == "La presión es 3500 psi"
            )
            assert len(fake_llm._call_log) == 1
            assert HISTORY_HEADER not in fake_llm._call_log[0]
        finally:
            self._stop_all(named_patches)

    def test_second_question_takes_prior_conversation_into_account(self):
        """Si ya hay preguntas anteriores en la conversación, el
        prompt de la siguiente pregunta debe incluirlas, verificado en el
        prompt real que efectivamente llega al LLM, no en un mock superficial."""
        fake_retriever = self._make_fake_retriever()
        fake_llm = self._make_fake_llm(
            responses=[
                ["La ", "presión ", "es 3500 psi"],
                ["Sí, ", "es la ", "misma presión"],
            ]
        )
        named_patches = self._patch_real_pipeline(fake_retriever, fake_llm)
        self._start_all(named_patches)
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            at.chat_input[0].set_value("¿Cuál es la presión del pozo CH-88?").run()
            at.chat_input[0].set_value("¿Y esa presión es la misma de siempre?").run()

            assert len(fake_llm._call_log) == 2
            second_prompt = fake_llm._call_log[1]

            assert HISTORY_HEADER in second_prompt
            assert "¿Cuál es la presión del pozo CH-88?" in second_prompt
            assert "La presión es 3500 psi" in second_prompt

            assert (
                at.session_state["messages"][-1]["content"] == "Sí, es la misma presión"
            )
        finally:
            self._stop_all(named_patches)

    def test_two_users_share_pipeline_but_not_history(self):
        """Dos sesiones distintas (dos
        usuarios) comparten la MISMA instancia de RAGPipeline real (cacheada
        con @st.cache_resource), pero cada una arma su propio prompt sin que
        el historial de una se filtre en el de la otra.

        A diferencia del test de UI equivalente (con RAGPipeline mockeado),
        acá se verifica que el aislamiento se sostiene incluso con
        RAGPipeline + PromptBuilder reales construyendo el prompt de verdad."""
        fake_retriever = self._make_fake_retriever()
        fake_llm = self._make_fake_llm(
            responses=[
                ["Respuesta ", "para A"],
                ["Respuesta ", "para B"],
                ["Segunda ", "respuesta ", "para A"],
            ]
        )
        named_patches = self._patch_real_pipeline(fake_retriever, fake_llm)
        self._start_all(named_patches)
        try:
            # Usuario A — primera pregunta
            at_a = AppTest.from_file(APP_PATH)
            at_a.run()
            at_a.chat_input[0].set_value("Pregunta de A").run()

            # Usuario B — sesión completamente distinta, primera pregunta
            at_b = AppTest.from_file(APP_PATH)
            at_b.run()
            at_b.chat_input[0].set_value("Pregunta de B").run()

            # Usuario A — segunda pregunta (debe recordar SU propio historial)
            at_a.chat_input[0].set_value("Segunda pregunta de A").run()

            assert len(fake_llm._call_log) == 3
            prompt_a1, prompt_b1, prompt_a2 = fake_llm._call_log

            # Primeras preguntas de cada usuario, sin historial cruzado
            assert HISTORY_HEADER not in prompt_a1
            assert HISTORY_HEADER not in prompt_b1

            # La segunda pregunta de A no tiene rastro de B
            assert "Pregunta de B" not in prompt_a2
            assert "Respuesta para B" not in prompt_a2

            # La segunda pregunta de A sí recuerda su propio turno anterior
            assert "Pregunta de A" in prompt_a2
            assert "Respuesta para A" in prompt_a2

            # Estado final en cada sesión: cada uno ve solo lo suyo
            assert [m["content"] for m in at_a.session_state["messages"]] == [
                "Pregunta de A",
                "Respuesta para A",
                "Segunda pregunta de A",
                "Segunda respuesta para A",
            ]
            assert [m["content"] for m in at_b.session_state["messages"]] == [
                "Pregunta de B",
                "Respuesta para B",
            ]
        finally:
            self._stop_all(named_patches)

    def test_pipeline_and_dependencies_built_once_across_users(self):
        """Aunque varios usuarios (sesiones) usen la app, Embedder,
        VectorStore, Retriever y LLMClient deben construirse una única vez
        gracias a @st.cache_resource, no una vez por sesión."""
        fake_retriever = self._make_fake_retriever()
        fake_llm = self._make_fake_llm(responses=[["ok para A"], ["ok para B"]])
        named_patches = self._patch_real_pipeline(fake_retriever, fake_llm)
        mocks = self._start_all(named_patches)
        try:
            at_a = AppTest.from_file(APP_PATH)
            at_a.run()
            at_a.chat_input[0].set_value("Pregunta de A").run()

            at_b = AppTest.from_file(APP_PATH)
            at_b.run()
            at_b.chat_input[0].set_value("Pregunta de B").run()

            # Cada constructor de infraestructura pesada se llamó una sola vez,
            # sin importar cuántas sesiones/usuarios usaron la app.
            assert mocks["embedder"].call_count == 1
            assert mocks["vectorstore"].call_count == 1
            assert mocks["retriever"].call_count == 1
            assert mocks["llm"].call_count == 1
        finally:
            self._stop_all(named_patches)
