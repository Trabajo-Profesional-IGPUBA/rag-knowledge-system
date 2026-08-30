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
        """Cubre CA-1.1, CA-1.2 (Historia 1) — input de chat y mensajes diferenciados por rol."""
        mock_pipeline, mock_llm, patches = self._patch_pipeline_internals(
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
        """Cubre CA-1.3 (Historia 1) — título y descripción visibles."""
        _, _, patches = self._patch_pipeline_internals()
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert "IGPUBA" in at.title[0].value
            assert "documentos técnicos" in at.caption[0].value.lower()
        finally:
            self._stop_patches(patches)

    def test_pipeline_initialized_once_and_reused(self):
        """Cubre CA-2.1 (Historia 2) — el pipeline se inicializa una sola vez y se reutiliza."""
        _, _, patches = self._patch_pipeline_internals(
            stream_tokens=["Respuesta", " 1"]
        )
        rag_pipeline_patch = patches[-1]  # el patch de src.llm.rag_pipeline.RAGPipeline
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()
            at.chat_input[0].set_value("pregunta 1").run()

            # Reinicia el mock de stream para la segunda interacción, sin tocar
            # el constructor de RAGPipeline: si se llamara de nuevo, cache_resource
            # habría fallado en su propósito.
            at.chat_input[0].set_value("pregunta 2").run()

            # RAGPipeline() como constructor debe haberse llamado una sola vez
            # gracias a @st.cache_resource, aunque hubo 2 interacciones de chat.
            rag_pipeline_ctor = rag_pipeline_patch  # el MagicMock del patch
        finally:
            self._stop_patches(patches)

    def test_pipeline_initialized_once_and_reused(self):
        """Cubre CA-2.1 (Historia 2) — el pipeline se inicializa una sola vez y se reutiliza."""
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
        """Cubre CA-2.3 (Historia 2) — error comprensible si falla la inicialización."""
        with patch(
            "src.embeddings.embedder.Embedder",
            side_effect=Exception("No se pudo cargar el vectorstore"),
        ):
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert len(at.error) > 0
            assert "Error al cargar el sistema" in at.error[0].value

    def test_blocks_chat_when_ollama_unavailable(self):
        """Cubre CA-3.1, CA-3.3 (Historia 3) — verifica disponibilidad y bloquea el chat si falla."""
        _, mock_llm, patches = self._patch_pipeline_internals(ollama_available=False)
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert len(at.chat_input) == 0
            mock_llm.is_available.assert_called_once()
        finally:
            self._stop_patches(patches)

    def test_shows_actionable_error_message(self):
        """Cubre CA-3.2 (Historia 3) — mensaje de error con acción concreta (comando)."""
        _, _, patches = self._patch_pipeline_internals(ollama_available=False)
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert len(at.error) > 0
            assert "ollama serve" in at.error[0].value
        finally:
            self._stop_patches(patches)

    def test_messages_persist_in_session_state(self):
        """Cubre CA-4.1, CA-4.2 (Historia 4) — mensajes se guardan y renderizan en orden."""
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
        """Cubre CA-4.3 (Historia 4) — historial vacío si no hay conversación previa."""
        _, _, patches = self._patch_pipeline_internals()
        try:
            at = AppTest.from_file(APP_PATH)
            at.run()

            assert at.session_state["messages"] == []
        finally:
            self._stop_patches(patches)

