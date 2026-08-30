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

  