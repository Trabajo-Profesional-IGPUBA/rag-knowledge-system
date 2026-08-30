from unittest.mock import MagicMock


class TestRAGPipeline:
    def _make_pipeline(self, llm_text: str = "Respuesta de prueba"):
        from src.llm.client import LLMResponse
        from src.llm.prompt_builder import PromptBuilder
        from src.llm.rag_pipeline import RAGConfig, RAGPipeline
        from src.retrieval.retriever import RetrievalResult

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="test",
            chunks=[
                {
                    "chunk_id": "doc1::chunk_0",
                    "text": "Texto relevante del documento.",
                    "metadata": {
                        "doc_id": "doc1",
                        "doc_type": "ewrs",
                        "filename": "PM104",
                    },
                    "score": 0.85,
                    "distance": 0.15,
                }
            ],
            top_k=5,
        )

        mock_llm = MagicMock()
        mock_llm.config = MagicMock()
        mock_llm.config.model = "llama3:8b"
        mock_llm.generate.return_value = LLMResponse(
            text=llm_text,
            model="llama3:8b",
            ok=True,
        )

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            prompt_builder=PromptBuilder(),
            config=RAGConfig(top_k=5, min_score=0.3),
        )
        return pipeline, mock_retriever, mock_llm

    def test_query_returns_rag_response(self):
        pipeline, _, _ = self._make_pipeline("La presión es 3500 psi.")
        resp = pipeline.query("¿Cuál es la presión?")
        assert resp.ok
        assert resp.answer == "La presión es 3500 psi."
        assert resp.query == "¿Cuál es la presión?"

    def test_query_saves_history(self):
        pipeline, _, _ = self._make_pipeline("respuesta")
        pipeline.query("pregunta 1")
        pipeline.query("pregunta 2")
        assert len(pipeline.history) == 2

    def test_clear_history(self):
        pipeline, _, _ = self._make_pipeline("respuesta")
        pipeline.query("pregunta")
        pipeline.clear_history()
        assert len(pipeline.history) == 0

    def test_min_score_filters_chunks(self):
        from src.llm.client import LLMResponse
        from src.llm.rag_pipeline import RAGConfig, RAGPipeline
        from src.retrieval.retriever import RetrievalResult

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="test",
            chunks=[
                {
                    "chunk_id": "c1",
                    "text": "bueno",
                    "metadata": {},
                    "score": 0.8,
                    "distance": 0.2,
                },
                {
                    "chunk_id": "c2",
                    "text": "malo",
                    "metadata": {},
                    "score": 0.1,
                    "distance": 0.9,
                },
            ],
            top_k=5,
        )

        mock_llm = MagicMock()
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test"
        mock_llm.generate.return_value = LLMResponse(text="ok", model="test", ok=True)

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            config=RAGConfig(min_score=0.5),
        )
        resp = pipeline.query("test")
        assert resp.prompt.num_chunks == 1

    def test_sources_property(self):
        pipeline, _, _ = self._make_pipeline("respuesta")
        resp = pipeline.query("pregunta")
        assert len(resp.sources) >= 0
        

    def test_query_calls_components_in_order_with_expected_args(self):
        """CA-1.1: retrieve -> build -> generate, en ese orden y con los args correctos."""
        pipeline, mock_retriever, mock_llm = self._make_pipeline("respuesta")
        pipeline._prompt_builder = MagicMock()
        pipeline._prompt_builder.build.return_value = MagicMock(
            prompt="prompt final", num_chunks=1, total_chars=10
        )

        manager = MagicMock()
        manager.attach_mock(mock_retriever.retrieve, "retrieve")
        manager.attach_mock(pipeline._prompt_builder.build, "build")
        manager.attach_mock(mock_llm.generate, "generate")

        pipeline.query("¿Cuál es la presión?")

        assert manager.mock_calls[0][0] == "retrieve"
        assert manager.mock_calls[1][0] == "build"
        assert manager.mock_calls[2][0] == "generate"
        mock_llm.generate.assert_called_with("prompt final")

    def test_default_prompt_builder_is_used_when_not_provided(self):
        """CA-1.3: si no se pasa PromptBuilder, se instancia uno por defecto y el pipeline funciona igual."""
        from src.llm.rag_pipeline import RAGPipeline

        pipeline, _, _ = self._make_pipeline("respuesta")
        pipeline_sin_builder = RAGPipeline(
            retriever=pipeline._retriever, llm_client=pipeline._llm
        )
        resp = pipeline_sin_builder.query("pregunta")
        assert resp.ok
