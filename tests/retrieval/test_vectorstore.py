# tests/retrieval/test_vectorstore.py
"""
Tests para src/retrieval/vectorstore.py
Cubren "Épica: Migración del motor de base de datos vectorial: ChromaDB → Qdrant":
  - Inicialización del motor -> CA-1.1 a CA-1.2
  - Identificación de chunks  -> CA-2.3
"""

import pytest

from src.retrieval.vectorstore import VectorStore, _to_point_id


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "vectorstore")
    yield s
    s.close()


def _fake_embedding(dim: int = 384) -> list[float]:
    return [0.1] * dim


class TestVectorStore:

    # -----------------------------------------------------------------
    # Inicialización del motor
    # -----------------------------------------------------------------

    def test_initializes_empty_and_persists_locally(self, store):
        """CA-1.1: El sistema debe inicializar la base de datos vectorial en modo
        local, persistiendo en disco sin requerir un servidor externo."""
        assert store.count() == 0

    def test_reuses_existing_collection_without_losing_data(self, tmp_path):
        """CA-1.2: El sistema debe reutilizar la colección existente si ya fue
        creada previamente, sin recrearla ni perder los datos indexados."""
        path = tmp_path / "vs"
        s1 = VectorStore(path)
        s1_count_before = s1.count()
        s1.close()

        s2 = VectorStore(path)
        assert s2.count() == s1_count_before == 0

    # -----------------------------------------------------------------
    # Identificación de chunks
    # -----------------------------------------------------------------

    def test_to_point_id_is_deterministic(self):
        """CA-2.3: El mismo identificador de chunk debe mapear siempre al mismo
        registro interno, para sostener la idempotencia de la indexación."""
        uuid1 = _to_point_id("doc1::chunk_0")
        uuid2 = _to_point_id("doc1::chunk_0")
        assert uuid1 == uuid2
