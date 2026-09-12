"""Tests para la interfaz de chat (app.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[2] / "app.py")


class TestChatApp:
    def setup_method(self):
        """Limpia el caché de st.cache_resource entre tests, porque load_pipeline
        usa @st.cache_resource y si no se limpia, la primera carga real queda
        cacheada globalmente y contamina los tests siguientes."""
        st.cache_resource.clear()

    def _patch_pipeline_internals(self, ollama_available=True, stream_tokens=None):
        """Mockea las clases internas que usa load_pipeline(), no la función en sí,
        porque AppTest ejecuta app.py en un contexto que no comparte la referencia
        parcheada de 'app.load_pipeline'."""
        mock_pipeline_instance = MagicMock()
        if stream_tokens is not None:
            mock_pipeline_instance.query_stream.return_value = iter(stream_tokens)

        mock_llm_instance = MagicMock()
        mock_llm_instance.is_available.return_value = ollama_available
        mock_llm_instance.config.model = "llama3:8b"

        patches = [
            patch("src.embeddings.embedder.Embedder", return_value=MagicMock()),
            patch("src.retrieval.vectorstore.VectorStore", return_value=MagicMock()),
            patch("src.retrieval.retriever.Retriever", return_value=MagicMock()),
            patch("src.llm.client.LLMClient", return_value=mock_llm_instance),
            patch(
                "src.llm.rag_pipeline.RAGPipeline", return_value=mock_pipeline_instance
            ),
        ]
        for p in patches:
            p.start()

        return mock_pipeline_instance, mock_llm_instance, patches

    def _stop_patches(self, patches):
        for p in patches:
            p.stop()

    # ---------- Historia 1: Chat con el usuario ----------

    def test_chat_input_and_role_display(self):
        """CA-1.1: El sistema debe permitir al usuario escribir una consulta en lenguaje natural
        a través de un campo de entrada de chat.
        CA-1.2: El sistema debe mostrar la consulta del usuario y la respuesta del asistente diferenciadas por rol (usuario / asistente)
        en formato de conversación."""
        _mock_pipeline, _mock_llm, patches = self._patch_pipeline_internals(
            stream_tokens=["Respuesta", " de prueba"]
        )
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()
            at.chat_input[0].set_value("¿Cuál es la presión del pozo CH-88?").run()

            assert at.session_state["messages"][0]["role"] == "user"
            assert (
                at.session_state["messages"][0]["content"]
                == "¿Cuál es la presión del pozo CH-88?"
            )
            assert at.session_state["messages"][1]["role"] == "assistant"
        finally:
            self._stop_patches(patches)

    def test_title_and_description_present(self):
        """CA-1.3: El sistema debe presentar un título y una descripción que identifiquen claramente el propósito del sistema
        (consulta de documentos técnicos de pozos)."""
        _, _, patches = self._patch_pipeline_internals()
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert "IGPUBA" in at.title[0].value
            assert "documentos técnicos" in at.caption[0].value.lower()
        finally:
            self._stop_patches(patches)

    def test_pipeline_initialized_once_and_reused(self):
        """CA-2.1: El sistema debe inicializar el pipeline RAG (embedder, vectorstore, retriever, cliente LLM)
        una sola vez y reutilizarlo entre interacciones, sin recargarlo en cada ejecución.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query_stream.side_effect = lambda q: iter(["ok"])
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.config.model = "llama3:8b"

        with (
            patch("src.embeddings.embedder.Embedder", return_value=MagicMock()),
            patch("src.retrieval.vectorstore.VectorStore", return_value=MagicMock()),
            patch("src.retrieval.retriever.Retriever", return_value=MagicMock()),
            patch("src.llm.client.LLMClient", return_value=mock_llm),
            patch(
                "src.llm.rag_pipeline.RAGPipeline", return_value=mock_pipeline
            ) as mock_rag_pipeline_ctor,
        ):
            at = AppTest.from_file(APP_PATH)
            at.run()
            at.chat_input[0].set_value("pregunta 1").run()
            at.chat_input[0].set_value("pregunta 2").run()

            # @st.cache_resource debe evitar que RAGPipeline() se construya
            # de nuevo en cada re-run del script.
            assert mock_rag_pipeline_ctor.call_count == 1

    def test_shows_error_on_init_failure(self):
        """CA-2.3: Si la inicialización falla, el sistema debe mostrar un mensaje de error comprensible
        y detener la ejecución de forma controlada, sin exponer un error técnico sin contexto.
        """
        with patch(
            "src.embeddings.embedder.Embedder",
            side_effect=Exception("No se pudo cargar el vectorstore"),
        ):
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert len(at.error) > 0
            assert "Error al cargar el sistema" in at.error[0].value

    def test_blocks_chat_when_ollama_unavailable(self):
        """CA-3.1: El sistema debe verificar si el servidor Ollama está disponible antes de habilitar el flujo de consulta.
        CA-3.3: Si Ollama no está disponible, el sistema no debe permitir continuar con la interacción de chat.
        """
        _, mock_llm, patches = self._patch_pipeline_internals(ollama_available=False)
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert len(at.chat_input) == 0
            mock_llm.is_available.assert_called_once()
        finally:
            self._stop_patches(patches)

    def test_shows_actionable_error_message(self):
        """CA-3.2: Si Ollama no está disponible, el sistema debe mostrar un mensaje de error
        que indique al usuario qué acción concreta tomar para resolverlo (por ejemplo, el comando a ejecutar).
        """
        _, _, patches = self._patch_pipeline_internals(ollama_available=False)
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert len(at.error) > 0
            assert "ollama serve" in at.error[0].value
        finally:
            self._stop_patches(patches)

    def test_messages_persist_in_session_state(self):
        """CA-4.1: El sistema debe almacenar cada mensaje (usuario y asistente)
        en el estado de sesión a medida que ocurre la conversación.
        CA-4.2: Al recargar o re-renderizar la interfaz, el sistema debe mostrar nuevamente todos los mensajes previamente almacenados,
        en el orden en que se generaron."""
        _, _, patches = self._patch_pipeline_internals(
            stream_tokens=["Respuesta", " completa"]
        )
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()
            at.chat_input[0].set_value("pregunta 1").run()

            assert len(at.session_state["messages"]) == 2
            assert at.session_state["messages"][0]["role"] == "user"
            assert at.session_state["messages"][1]["role"] == "assistant"
        finally:
            self._stop_patches(patches)

    def test_history_starts_empty(self):
        """CA-4.3: El historial debe iniciar vacío cuando no existe una conversación previa en la sesión."""
        _, _, patches = self._patch_pipeline_internals()
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert at.session_state["messages"] == []
        finally:
            self._stop_patches(patches)

    def test_streaming_response_progressive(self):
        """CA-5.2: Al enviar una consulta, el sistema debe mostrar la respuesta del asistente de forma progresiva,
        token a token, a medida que se va generando."""
        mock_pipeline, _, patches = self._patch_pipeline_internals(
            stream_tokens=["La ", "presión ", "es 3500 psi"]
        )
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()
            at.chat_input[0].set_value("¿presión?").run()

            final_message = at.session_state["messages"][-1]["content"]
            assert final_message == "La presión es 3500 psi"
            mock_pipeline.query_stream.assert_called_once_with("¿presión?", history=[])
        finally:
            self._stop_patches(patches)

    def test_cursor_appears_during_streaming_and_disappears_after(self):
        """CA-6.1: Mientras la respuesta se genera token a token, el sistema debe mostrar un indicador visual (cursor)
        al final del texto parcial ya generado.
        CA-6.2: El cursor debe dejar de mostrarse una vez que la generación de la respuesta finaliza.
        CA-6.3: El texto final mostrado, sin el cursor, debe coincidir exactamente con la respuesta completa generada.
        """
        from app import render_streaming_response

        mock_placeholder = MagicMock()
        tokens = iter(["Hola", " mundo"])

        result = render_streaming_response(mock_placeholder, tokens)

        calls = [c.args[0] for c in mock_placeholder.markdown.call_args_list]
        assert calls[0] == "Hola▌"
        assert calls[1] == "Hola mundo▌"
        assert calls[-1] == "Hola mundo"
        assert result == "Hola mundo"
