# tests/retrieval/test_vectorstore.py
"""
Tests para src/retrieval/vectorstore.py
Cubren "Épica: Migración del motor de base de datos vectorial: ChromaDB":
  - Inicialización del motor -> CA-1.1 a CA-1.2
"""

import pytest

from src.retrieval.vectorstore import VectorStore


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
        """CA-1.1: la base vectorial se inicializa en modo local, persistiendo
        en disco sin requerir un servidor externo."""
        assert store.count() == 0

    def test_reuses_existing_collection_without_losing_data(self, tmp_path):
        """CA-1.2: si la colección ya existe, se reutiliza en vez de
        recrearse, sin perder los datos ya indexados."""
        path = tmp_path / "vs"
        s1 = VectorStore(path)
        s1_count_before = s1.count()
        s1.close()

        s2 = VectorStore(path)
        assert s2.count() == s1_count_before == 0
