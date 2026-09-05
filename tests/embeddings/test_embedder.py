from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.embedder import DEFAULT_MODEL, EMBEDDING_DIM, Embedder

# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder():
    """Embedder con modelo mockeado para no descargar pesos en tests."""
    with patch("src.embeddings.embedder.SentenceTransformer") as MockST:
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

        mock_embedder._model.encode.return_value = np.ones(
            (3, EMBEDDING_DIM), dtype="float32"
        )  # batch necesita 2D
        result = mock_embedder.embed_batch(["a", "b", "c"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], list)

    def test_get_dimension(self, mock_embedder):
        assert mock_embedder.get_dimension() == EMBEDDING_DIM

    def test_embed_values_are_floats(self, mock_embedder):
        result = mock_embedder.embed("texto")
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Implementación del servicio de embeddings
# ---------------------------------------------------------------------------


def test_can_specify_which_model_to_load_at_init():
    """CA-3.1: El sistema debe permitir indicar qué modelo cargar al inicializar el servicio de embeddings, usando el modelo por defecto si no se especifica otro."""
    with patch("src.embeddings.embedder.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
        MockST.return_value = mock_model
        Embedder("modelo-custom")
        MockST.assert_called_once_with("modelo-custom")


def test_uses_default_model_when_not_specified():
    """CA-3.1: El sistema debe permitir indicar qué modelo cargar al inicializar el servicio de embeddings, usando el modelo por defecto si no se especifica otro."""
    with patch("src.embeddings.embedder.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
        MockST.return_value = mock_model
        Embedder()
        MockST.assert_called_once_with(DEFAULT_MODEL)


def test_model_loaded_once_and_reused_across_calls(mock_embedder):
    """CA-3.2: El modelo debe cargarse una única vez al inicializar el servicio, y esa misma instancia debe reutilizarse en todas las generaciones de embeddings posteriores, sin recargar el modelo en cada uso."""
    mock_embedder.embed("a")
    mock_embedder.embed("b")
    mock_embedder._model.encode.return_value = np.ones(
        (2, EMBEDDING_DIM), dtype="float32"
    )
    mock_embedder.embed_batch(["c", "d"])
    assert mock_embedder._model.encode.call_count == 3
