from unittest.mock import MagicMock

import pytest

from src.retrieval.retriever import RetrievalResult, Retriever, evaluate_topk_accuracy

EMBEDDING_DIM = 768


def _fake_embedding():
    return [0.1] * EMBEDDING_DIM


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


class TestEvaluateTopKAccuracy:
    def test_perfect_accuracy(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(
            chunks=[{"metadata": {"doc_id": "doc1"}}, {"metadata": {"doc_id": "doc2"}}]
        )
        queries = [
            ("query1", ["doc1"]),
            ("query2", ["doc1"]),
        ]  # ambas esperan doc1 que siempre está en top-1
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
