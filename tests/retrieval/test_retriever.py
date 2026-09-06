from unittest.mock import MagicMock

import pytest

from src.retrieval.retriever import RetrievalResult, Retriever, evaluate_topk_accuracy


def _fake_embedding():
    return [0.1] * 384


@pytest.fixture
def mock_retriever():
    embedder = MagicMock()
    embedder.embed.return_value = _fake_embedding()

    vectorstore = MagicMock()
    vectorstore.search.return_value = [
        {
            "chunk_id": "doc1::chunk_0",
            "text": "texto relevante",
            "metadata": {"doc_id": "doc1", "doc_type": "ewrs"},
            "score": 0.92,
            "distance": 0.08,
        },
        {
            "chunk_id": "doc2::chunk_0",
            "text": "otro texto",
            "metadata": {"doc_id": "doc2", "doc_type": "parte_diario"},
            "score": 0.75,
            "distance": 0.25,
        },
    ]

    return Retriever(embedder=embedder, vectorstore=vectorstore)


class TestRetriever:

    def test_retrieve_returns_result(self, mock_retriever):
        result = mock_retriever.retrieve("¿pérdida de circulación en Quintuco?")
        assert isinstance(result, RetrievalResult)

    def test_retrieve_calls_embedder(self, mock_retriever):
        mock_retriever.retrieve("consulta")
        mock_retriever._embedder.embed.assert_called_once_with("consulta")

    def test_retrieve_chunks_count(self, mock_retriever):
        result = mock_retriever.retrieve("consulta", top_k=2)
        assert len(result.chunks) == 2

    def test_result_texts_property(self, mock_retriever):
        result = mock_retriever.retrieve("consulta")
        assert result.texts == ["texto relevante", "otro texto"]

    def test_result_best_score(self, mock_retriever):
        result = mock_retriever.retrieve("consulta")
        assert result.best_score == 0.92

    def test_retrieve_passes_filters(self, mock_retriever):
        mock_retriever.retrieve("consulta", filters={"doc_type": "ewrs"})
        mock_retriever._vectorstore.search.assert_called_once()
        call_kwargs = mock_retriever._vectorstore.search.call_args[1]
        assert call_kwargs.get("filters") == {"doc_type": "ewrs"}

    def test_ca11_query_is_embedded_before_search(self, mock_retriever):
        """CA-1.1: La consulta debe convertirse a embedding antes de buscar."""
        mock_retriever.retrieve("pérdida de circulación en Quintuco")
        mock_retriever._embedder.embed.assert_called_once_with(
            "pérdida de circulación en Quintuco"
        )

    def test_ca11_embedding_passed_to_vectorstore(self, mock_retriever):
        """CA-1.1: El embedding de la consulta debe pasarse al vectorstore para la búsqueda."""
        mock_retriever.retrieve("consulta técnica")
        call_kwargs = mock_retriever._vectorstore.search.call_args[1]
        assert call_kwargs["query_embedding"] == _fake_embedding()

    def test_ca12_embedder_error_propagates_clearly(self):
        """CA-1.2: Si el embedding falla, el error debe propagarse de forma clara."""
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("fallo en modelo")
        vectorstore = MagicMock()
        retriever = Retriever(embedder=embedder, vectorstore=vectorstore)

        with pytest.raises(RuntimeError, match="fallo en modelo"):
            retriever.retrieve("consulta")

    def test_ca21_returns_top_k_chunks(self, mock_retriever):
        """CA-2.1: El sistema debe recuperar exactamente K chunks ordenados por score."""
        result = mock_retriever.retrieve("consulta", top_k=2)
        assert len(result.chunks) == 2

    def test_ca21_chunks_ordered_by_score_descending(self, mock_retriever):
        """CA-2.1: Los chunks deben estar ordenados de mayor a menor score."""
        result = mock_retriever.retrieve("consulta", top_k=2)
        scores = [c["score"] for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_ca21_top_k_passed_to_vectorstore(self, mock_retriever):
        """CA-2.1: El valor de top_k debe pasarse correctamente al vectorstore."""
        mock_retriever.retrieve("consulta", top_k=3)
        call_kwargs = mock_retriever._vectorstore.search.call_args[1]
        assert call_kwargs["n_results"] == 3

    def test_ca22_filters_passed_to_vectorstore(self, mock_retriever):
        """CA-2.2: Los filtros de metadata deben pasarse al vectorstore."""
        mock_retriever.retrieve("consulta", filters={"doc_type": "parte_diario"})
        call_kwargs = mock_retriever._vectorstore.search.call_args[1]
        assert call_kwargs.get("filters") == {"doc_type": "parte_diario"}

    def test_ca22_no_filters_when_not_specified(self, mock_retriever):
        """CA-2.2: Si no se especifican filtros, no deben pasarse al vectorstore."""
        mock_retriever.retrieve("consulta")
        call_kwargs = mock_retriever._vectorstore.search.call_args[1]
        assert call_kwargs.get("filters") is None

    def test_ca23_each_result_has_score(self, mock_retriever):
        """CA-2.3: Cada chunk recuperado debe incluir su score de similitud."""
        result = mock_retriever.retrieve("consulta")
        for chunk in result.chunks:
            assert "score" in chunk
            assert isinstance(chunk["score"], float)

    def test_ca23_best_score_property(self, mock_retriever):
        """CA-2.3: La propiedad best_score debe retornar el score del chunk más relevante."""
        result = mock_retriever.retrieve("consulta")
        assert result.best_score == 0.92

    def test_ca23_best_score_zero_when_no_chunks(self):
        """CA-2.3: Si no hay chunks recuperados, best_score debe ser 0."""
        embedder = MagicMock()
        embedder.embed.return_value = _fake_embedding()
        vectorstore = MagicMock()
        vectorstore.search.return_value = []
        retriever = Retriever(embedder=embedder, vectorstore=vectorstore)
        result = retriever.retrieve("consulta sin resultados")
        assert result.best_score == 0.0

    def test_ca31_each_result_has_text(self, mock_retriever):
        """CA-3.1: Cada resultado debe incluir el texto del chunk."""
        result = mock_retriever.retrieve("consulta")
        for chunk in result.chunks:
            assert "text" in chunk
            assert isinstance(chunk["text"], str)

    def test_ca31_each_result_has_metadata(self, mock_retriever):
        """CA-3.1: Cada resultado debe incluir la metadata del documento de origen."""
        result = mock_retriever.retrieve("consulta")
        for chunk in result.chunks:
            assert "metadata" in chunk
            assert "doc_id" in chunk["metadata"]

    def test_ca32_result_exposes_chunk_count(self, mock_retriever):
        """CA-3.2: El resultado debe exponer cuántos chunks fueron recuperados."""
        result = mock_retriever.retrieve("consulta", top_k=2)
        assert len(result.chunks) == 2

    def test_ca32_texts_property_matches_chunks(self, mock_retriever):
        """CA-3.2: La propiedad texts debe tener la misma cantidad que chunks."""
        result = mock_retriever.retrieve("consulta")
        assert len(result.texts) == len(result.chunks)


class TestEvaluateTopKAccuracy:

    def test_perfect_accuracy(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(
            chunks=[{"metadata": {"doc_id": "doc1"}}, {"metadata": {"doc_id": "doc2"}}]
        )
        queries = [("query1", ["doc1"]), ("query2", ["doc1"])]
        acc = evaluate_topk_accuracy(retriever, queries, k_values=[1, 2])
        assert acc["top_1"] == 1.0

    def test_zero_accuracy(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(
            chunks=[{"metadata": {"doc_id": "doc_otro"}}]
        )
        queries = [("query", ["doc_esperado"])]
        acc = evaluate_topk_accuracy(retriever, queries, k_values=[1])
        assert acc["top_1"] == 0.0

    def test_empty_queries_returns_zeros(self):
        retriever = MagicMock()
        acc = evaluate_topk_accuracy(retriever, [], k_values=[1, 3, 5])
        assert all(v == 0.0 for v in acc.values())
