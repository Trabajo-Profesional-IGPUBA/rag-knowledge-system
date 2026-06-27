from unittest.mock import MagicMock, patch
import pytest
from src.embeddings.embedder import Embedder, EMBEDDING_DIM, DEFAULT_MODEL


@pytest.fixture
def mock_embedder():
    """Embedder con modelo mockeado para no descargar en tests."""
    with patch("src.embeddings.embedder.SentenceTransformer") as MockST:
        import numpy as np
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
        mock_model.encode.return_value = np.ones(EMBEDDING_DIM, dtype="float32")
        MockST.return_value = mock_model
        yield Embedder(DEFAULT_MODEL)


class TestEmbedder:
    def test_embed_returns_list(self, mock_embedder):
        result = mock_embedder.embed("texto de prueba")
        assert isinstance(result, list)

    def test_embed_returns_correct_dimension(self, mock_embedder):
        result = mock_embedder.embed("texto")
        assert len(result) == EMBEDDING_DIM

    def test_embed_batch_empty_returns_empty(self, mock_embedder):
        result = mock_embedder.embed_batch([])
        assert result == []

    def test_embed_batch_returns_list_of_lists(self, mock_embedder):
        import numpy as np
        mock_embedder._model.encode.return_value = np.ones((3, EMBEDDING_DIM), dtype="float32")  # batch necesita 2D
        result = mock_embedder.embed_batch(["a", "b", "c"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], list)

    def test_get_dimension(self, mock_embedder):
        assert mock_embedder.get_dimension() == EMBEDDING_DIM

    def test_embed_values_are_floats(self, mock_embedder):
        result = mock_embedder.embed("texto")
        assert all(isinstance(v, float) for v in result)
