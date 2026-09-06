"""
Tests para src/embeddings/embedder.py

Cubren "Integración del modelo seleccionado":
  - Implementación del servicio de embeddings -> CA-3.1 a CA-3.7
  - Validación inicial del funcionamiento  -> CA-4.1, CA-4.2, CA-4.3

El modelo real (SentenceTransformer) se mockea en todos los casos para no
depender de descargar pesos ni de tiempos de carga reales.
"""

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


def test_can_query_currently_loaded_model(mock_embedder):
    """CA-3.3: El servicio de embeddings debe exponer el modelo actualmente cargado."""
    assert mock_embedder.model_name == DEFAULT_MODEL
    custom_embedder = Embedder(model_name="un-modelo-custom")
    assert custom_embedder.model_name == "un-modelo-custom"


def test_vectorizing_single_text_returns_list_of_floats(mock_embedder):
    """CA-3.4: Al vectorizar un texto individual, el sistema debe devolver el embedding resultante como una lista de números de punto flotante."""
    result = mock_embedder.embed("texto de prueba")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_vectorizing_list_of_texts_preserves_order(mock_embedder):
    """CA-3.5: Al vectorizar una lista de textos, el sistema debe devolver una lista de embeddings, uno por cada texto, respetando el mismo orden en que fueron entregados."""
    vectors = np.stack([np.full(EMBEDDING_DIM, i, dtype="float32") for i in range(3)])
    mock_embedder._model.encode.return_value = vectors
    result = mock_embedder.embed_batch(["a", "b", "c"])
    assert len(result) == 3
    assert result[0][0] == 0.0
    assert result[1][0] == 1.0
    assert result[2][0] == 2.0


def test_batch_size_and_progress_flag_are_configurable(mock_embedder):
    """CA-3.6: El sistema debe permitir configurar el tamaño de lote y si se muestra o no una barra de progreso al vectorizar múltiples textos, y aplicar esa configuración durante la generación."""
    mock_embedder._model.encode.return_value = np.ones(
        (2, EMBEDDING_DIM), dtype="float32"
    )
    mock_embedder.embed_batch(["a", "b"], batch_size=8, show_progress=True)
    _, kwargs = mock_embedder._model.encode.call_args
    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is True


def test_default_batch_size_applied_when_not_specified(mock_embedder):
    """CA-3.6: El sistema debe permitir configurar el tamaño de lote y si se muestra o no una barra de progreso al vectorizar múltiples textos, y aplicar esa configuración durante la generación."""
    mock_embedder._model.encode.return_value = np.ones(
        (1, EMBEDDING_DIM), dtype="float32"
    )
    mock_embedder.embed_batch(["a"])
    _, kwargs = mock_embedder._model.encode.call_args
    assert kwargs["batch_size"] == 32


def test_vectorizing_empty_list_returns_empty_without_generating(mock_embedder):
    """CA-3.7: Al vectorizar una lista vacía de textos, el sistema debe devolver una lista vacía sin ejecutar ninguna generación de embeddings."""
    result = mock_embedder.embed_batch([])
    assert result == []
    mock_embedder._model.encode.assert_not_called()


# ---------------------------------------------------------------------------
# Validación inicial del funcionamiento
# ---------------------------------------------------------------------------


def test_queried_dimension_reflects_loaded_model_not_a_fixed_value(mock_embedder):
    """CA-4.1: Al consultar la dimensión de los embeddings, el sistema debe devolver la dimensión real reportada por el modelo actualmente cargado, no un valor fijo definido de antemano."""
    mock_embedder._model.get_sentence_embedding_dimension.return_value = 999
    assert mock_embedder.get_dimension() == 999


def test_vector_has_expected_number_of_components_with_default_model(mock_embedder):
    """CA-4.2: Al vectorizar un texto con el modelo por defecto, el vector resultante debe tener exactamente la cantidad de componentes definida como dimensión esperada del sistema."""
    result = mock_embedder.embed("texto")
    assert len(result) == EMBEDDING_DIM


def test_embedding_components_are_native_python_floats(mock_embedder):
    """CA-4.3: Cada componente de un embedding generado, ya sea individual o en lote, debe quedar representado como número de punto flotante nativo, de forma que pueda serializarse correctamente a JSON o ser almacenado en la base vectorial."""
    single = mock_embedder.embed("texto")
    assert all(isinstance(v, float) and not isinstance(v, np.floating) for v in single)

    mock_embedder._model.encode.return_value = np.ones(
        (2, EMBEDDING_DIM), dtype="float32"
    )
    batch = mock_embedder.embed_batch(["a", "b"])
    assert all(
        isinstance(v, float) and not isinstance(v, np.floating)
        for vec in batch
        for v in vec
    )
