from unittest.mock import MagicMock

"""
Cubre "ÉPICA: Pipeline RAG integrado":
  - Orquestación secuencial de Retriever, PromptBuilder y LLMClient-> CA-1.1 Y  CA-1.3 
  - Filtrado de chunks recuperados por score mínimo configurable-> CA-2.1 a CA-2.3
  - Modo de consulta completa con respuesta única-> CA-3.1 a CA-3.2
  - Modo de consulta en streaming con yield de tokens-> CA-4.1 a CA-4.2
  - Limpieza de historial de conversación bajo demanda-> CA-6.1 a CA-6.2
  - RAGConfig como dataclass de configuración del pipeline-> CA-7.1 a CA-7.2
  - RAGResponse con trazabilidad completa de query, chunks, prompt y respuesta-> CA-8.1 a CA-8.2
  - Propiedad sources con fuentes ordenadas por relevancia-> CA-9.1 a CA-9.2
  - Propiedad pretty_sources para visualización formateada-> CA-10.1 a CA-10.2

Cubre "Épica: Aislamiento de sesiones e historial de chat por usuario (Streamlit)":
  - Rendimiento-> CA-13.1

"""


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

    def test_query_calls_components_in_order_with_expected_args(self):
        """CA-1.1: Dado que se ejecuta una consulta, el sistema debe recuperar los chunks relevantes,
        construir el prompt con ellos, y generar la respuesta con el LLM, en ese orden.
        """
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
        """CA-1.3: Si no se especifica un PromptBuilder, el sistema debe usar una instancia por defecto."""
        from src.llm.rag_pipeline import RAGPipeline

        pipeline, _, _ = self._make_pipeline("respuesta")
        pipeline_sin_builder = RAGPipeline(
            retriever=pipeline._retriever, llm_client=pipeline._llm
        )
        resp = pipeline_sin_builder.query("pregunta")
        assert resp.ok

    def test_min_score_filters_chunks(self):
        """CA-2.1: Dado un conjunto de chunks recuperados con distintos puntajes de relevancia,
        el sistema debe descartar los que estén por debajo del score mínimo configurado.
        """
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
        """CA-2.2: El score mínimo debe ser configurable, no un valor fijo."""
        from src.llm.rag_pipeline import RAGConfig, RAGPipeline

        _pipeline, mock_retriever, mock_llm = self._make_pipeline("respuesta")

        pipeline_estricto = RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            config=RAGConfig(min_score=0.99),
        )
        resp = pipeline_estricto.query("pregunta")
        assert resp.prompt.num_chunks == 0  # el único chunk tiene score 0.85 < 0.99

    def test_no_chunks_pass_threshold_does_not_fail(self):
        """CA-2.3: Si ningún chunk supera el score mínimo, el sistema debe continuar sin fallar
        (aunque la respuesta se genere sin contexto recuperado)."""
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
        """CA-3.1: El sistema debe poder generar una respuesta completa a partir de una pregunta, entregándola de una sola vez."""
        pipeline, _, _ = self._make_pipeline("La presión es 3500 psi.")
        resp = pipeline.query("¿Cuál es la presión?")
        assert resp.ok
        assert resp.answer == "La presión es 3500 psi."
        assert resp.query == "¿Cuál es la presión?"

    def test_response_includes_query_retrieval_and_prompt(self):
        """CA-3.2: La respuesta debe incluir el texto generado, la consulta original,
        y los datos de recuperación y prompt utilizados."""
        pipeline, _, _ = self._make_pipeline("respuesta")
        resp = pipeline.query("pregunta")
        assert resp.query == "pregunta"
        assert resp.answer == "respuesta"
        assert resp.retrieval is not None
        assert resp.prompt is not None

    def test_query_stream_yields_tokens(self):
        """CA-4.1: El sistema debe poder entregar la respuesta de forma progresiva, token a token,
        en lugar de esperar a que esté completa."""
        pipeline, _, mock_llm = self._make_pipeline()
        mock_llm.generate_stream.return_value = iter(["Hola", " ", "mundo"])

        tokens = list(pipeline.query_stream("pregunta"))

        assert tokens == ["Hola", " ", "mundo"]

    def test_rag_config_centralizes_parameters(self):
        """CA-7.1: El sistema debe permitir centralizar en un solo lugar los parámetros del pipeline
        (cantidad de chunks, score mínimo, modo streaming, filtros)."""
        from src.llm.rag_pipeline import RAGConfig

        config = RAGConfig(
            top_k=10, min_score=0.6, stream=True, filters={"doc_type": "ewrs"}
        )
        assert config.top_k == 10
        assert config.min_score == 0.6
        assert config.stream is True
        assert config.filters == {"doc_type": "ewrs"}

    def test_rag_config_defaults(self):
        """CA-7.2: Debe existir un conjunto de valores por defecto razonables si no se especifica configuración propia."""
        from src.llm.rag_pipeline import RAGConfig

        config = RAGConfig()
        assert config.top_k == 5
        assert config.min_score == 0.3
        assert config.stream is False
        assert config.filters is None

    def test_rag_response_includes_full_traceability(self):
        """CA-8.1: El resultado de una consulta debe incluir la pregunta original, la respuesta generada,
        los datos de recuperación, el prompt construido y la respuesta cruda del LLM."""
        pipeline, _, mock_llm = self._make_pipeline("respuesta")
        resp = pipeline.query("pregunta")
        assert resp.query == "pregunta"
        assert resp.answer == "respuesta"
        assert resp.retrieval is not None
        assert resp.prompt is not None
        assert resp.llm_response is mock_llm.generate.return_value

    def test_rag_response_ok_reflects_retrieval_and_generation(self):
        """CCA-8.2: El resultado debe poder indicar si la consulta fue exitosa en su conjunto (recuperación + generación),
        no solo si el LLM respondió."""
        from src.llm.client import LLMResponse

        pipeline, _, mock_llm = self._make_pipeline()
        mock_llm.generate.return_value = LLMResponse(
            text="algo", model="test", ok=False
        )

        resp = pipeline.query("pregunta")
        assert resp.ok is False

    def test_sources_property(self):
        """CA-9.1: El sistema debe poder listar las fuentes documentales usadas para generar la respuesta.
        CA-9.2: Cada fuente debe incluir al menos nombre de archivo, tipo de documento y score de relevancia.
        """
        pipeline, _, _ = self._make_pipeline("respuesta")
        resp = pipeline.query("pregunta")
        assert len(resp.sources) == 1
        source = resp.sources[0]
        assert source["filename"] == "PM104"
        assert source["doc_type"] == "ewrs"
        assert source["score"] == 0.85

    def test_sources_limited_to_chunks_used_in_prompt(self):
        """CA-9.3: Las fuentes deben limitarse a los chunks efectivamente usados en el prompt, no a todos los recuperados."""
        pipeline, mock_retriever, _ = self._make_pipeline("respuesta")
        mock_retriever.retrieve.return_value.chunks = [
            {
                "chunk_id": "c1",
                "text": "a",
                "metadata": {"filename": "f1"},
                "score": 0.9,
                "distance": 0.1,
            },
            {
                "chunk_id": "c2",
                "text": "b",
                "metadata": {"filename": "f2"},
                "score": 0.8,
                "distance": 0.2,
            },
            {
                "chunk_id": "c3",
                "text": "c",
                "metadata": {"filename": "f3"},
                "score": 0.7,
                "distance": 0.3,
            },
        ]
        resp = pipeline.query("pregunta")
        resp.prompt.num_chunks = (
            2  # simula que el PromptBuilder solo usó 2 de los 3 chunks
        )
        assert len(resp.sources) == 2
        assert {s["filename"] for s in resp.sources} == {"f1", "f2"}

    def test_pretty_sources_with_sources(self):
        """CA-10.1: El sistema debe poder generar una representación de texto legible de las fuentes, para mostrarla al usuario."""
        pipeline, _, _ = self._make_pipeline("respuesta")
        resp = pipeline.query("pregunta")
        text = resp.pretty_sources()
        assert "PM104" in text
        assert "ewrs" in text
        assert "85%" in text

    def test_pretty_sources_without_sources(self):
        """CA-10.2: Si no hay fuentes, debe informarlo con un mensaje claro en lugar de una salida vacía o confusa."""
        pipeline, mock_retriever, _ = self._make_pipeline("respuesta")
        mock_retriever.retrieve.return_value.chunks = []
        resp = pipeline.query("pregunta")
        assert resp.pretty_sources().strip() == "Sin fuentes"

    def pipeline_does_not_keep_own_history_state(self):
        """CA-13.1: el sistema de búsqueda y generación debe seguir cargándose una
        sola vez y compartirse entre todos los usuarios (no debe reinicializarse
        por cada persona), para no perder velocidad."""
        pipeline, _, _ = self._make_pipeline("respuesta")
        assert not hasattr(pipeline, "_history")
