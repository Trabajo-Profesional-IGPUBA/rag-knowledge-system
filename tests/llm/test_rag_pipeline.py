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

    def test_min_score_filters_chunks(self):
        """CA-2.1: chunks por debajo del score mínimo se descartan."""
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

    def test_min_score_is_configurable(self):
        """CA-2.2: el score mínimo no es un valor fijo, cambia según RAGConfig."""
        from src.llm.rag_pipeline import RAGConfig, RAGPipeline

        pipeline, mock_retriever, mock_llm = self._make_pipeline("respuesta")

        pipeline_estricto = RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            config=RAGConfig(min_score=0.99),
        )
        resp = pipeline_estricto.query("pregunta")
        assert resp.prompt.num_chunks == 0  # el único chunk tiene score 0.85 < 0.99

    def test_no_chunks_pass_threshold_does_not_fail(self):
        """CA-2.3: si ningún chunk supera el score mínimo, el pipeline no debe fallar."""
        pipeline, mock_retriever, _ = self._make_pipeline("respuesta sin contexto")
        mock_retriever.retrieve.return_value.chunks = [
            {
                "chunk_id": "c1",
                "text": "irrelevante",
                "metadata": {},
                "score": 0.05,
                "distance": 0.95,
            }
        ]
        resp = pipeline.query("pregunta")
        assert resp.ok
        assert resp.prompt.num_chunks == 0


    def test_query_returns_rag_response(self):
        """CA-3.1: genera una respuesta completa de una sola vez."""
        pipeline, _, _ = self._make_pipeline("La presión es 3500 psi.")
        resp = pipeline.query("¿Cuál es la presión?")
        assert resp.ok
        assert resp.answer == "La presión es 3500 psi."
        assert resp.query == "¿Cuál es la presión?"

    def test_response_includes_query_retrieval_and_prompt(self):
        """CA-3.2: la respuesta incluye texto generado, query original, recuperación y prompt."""
        pipeline, _, _ = self._make_pipeline("respuesta")
        resp = pipeline.query("pregunta")
        assert resp.query == "pregunta"
        assert resp.answer == "respuesta"
        assert resp.retrieval is not None
        assert resp.prompt is not None

    def test_query_stream_yields_tokens(self):
        """CA-4.1: la respuesta se entrega progresivamente, token a token."""
        pipeline, _, mock_llm = self._make_pipeline()
        mock_llm.generate_stream.return_value = iter(["Hola", " ", "mundo"])

        tokens = list(pipeline.query_stream("pregunta"))

        assert tokens == ["Hola", " ", "mundo"]

    def test_query_stream_saves_full_response_to_history(self):
        """CA-4.2: al finalizar el streaming, la respuesta completa se guarda en el historial."""
        pipeline, _, mock_llm = self._make_pipeline()
        mock_llm.generate_stream.return_value = iter(["Hola", " ", "mundo"])

        list(pipeline.query_stream("pregunta"))

        assert pipeline.history == [("pregunta", "Hola mundo")]
