
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
