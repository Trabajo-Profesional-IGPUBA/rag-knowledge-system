"""
Test de integración para src/retrieval/vectorstore.py

A diferencia de test_vectorstore.py, este archivo ejercita el flujo real end-to-end contra Qdrant embebido,
sin mockear nada: simula el ciclo de vida completo tal como lo usaría el pipeline de indexación y recuperación en producción.
"""

from src.retrieval.vectorstore import VectorStore

EMBEDDING_DIM = 768


def _direction_a(dim: int = EMBEDDING_DIM) -> list[float]:
    """Vector base: dirección A."""
    return [1.0, 0.0] * (dim // 2)


def _direction_a_close(dim: int = EMBEDDING_DIM) -> list[float]:
    """Vector muy cercano a la dirección A (alta similitud coseno)."""
    return [0.95, 0.05] * (dim // 2)


def _direction_b_orthogonal(dim: int = EMBEDDING_DIM) -> list[float]:
    """Vector ortogonal a A (similitud coseno ≈ 0)."""
    return [0.0, 1.0] * (dim // 2)


class TestVectorStoreIntegration:

    def test_full_lifecycle_index_search_delete_reset_persist_close(self, tmp_path):
        """
        Flujo completo:
          1. Se inicializa el store en disco.
          2. Se indexa un lote mixto (dos doc_type distintos).
          3. Se busca con filtro y se valida ranking + contrato de salida.
          4. Se reprocesa un doc (delete_by_doc + reindex) sin duplicar.
          5. Los datos persisten al abrir una nueva instancia sobre el
             mismo path (simulando un proceso nuevo).
          6. Se reinicia el índice completo (cambio de modelo de embeddings).
          7. Se cierra la conexión explícitamente sin errores.
        """
        path = tmp_path / "vectorstore_integration"

        # 1. Inicialización
        store = VectorStore(path)
        assert store.count() == 0

        query_vector = _direction_a()
        close_vector = _direction_a_close()
        far_vector = _direction_b_orthogonal()

        # 2. Indexación de un lote mixto
        store.add_batch(
            chunk_ids=["ewr_001::chunk_0", "ewr_001::chunk_1", "parte_002::chunk_0"],
            texts=[
                "Pérdida de circulación en Quintuco a 2500m.",
                "Se detecta gas en superficie, monitoreo continuo.",
                "Parte diario: turno noche sin novedades.",
            ],
            embeddings=[close_vector, far_vector, far_vector],
            metadatas=[
                {"doc_id": "ewr_001", "doc_type": "ewrs", "chunk_index": 0},
                {"doc_id": "ewr_001", "doc_type": "ewrs", "chunk_index": 1},
                {"doc_id": "parte_002", "doc_type": "parte_diario", "chunk_index": 0},
            ],
        )
        assert store.count() == 3

        # 3. Búsqueda semántica con filtro
        results = store.search(query_vector, n_results=5, filters={"doc_type": "ewrs"})
        assert len(results) == 2
        # chunk_0 (close_vector) es inequívocamente más similar a query_vector
        # que chunk_1 (far_vector, ortogonal) — no hay empate de score posible.
        assert results[0]["chunk_id"] == "ewr_001::chunk_0"
        for key in ("chunk_id", "text", "metadata", "distance", "score"):
            assert key in results[0]
        assert all(r["metadata"]["doc_type"] == "ewrs" for r in results)

        # 4. Reprocesamiento de un documento (simula corrección de contenido)
        store.delete_by_doc("ewr_001")
        assert store.count() == 1  # solo queda parte_002

        store.add(
            "ewr_001::chunk_0",
            "Pérdida de circulación corregida y ampliada.",
            close_vector,
            {"doc_id": "ewr_001", "doc_type": "ewrs", "chunk_index": 0},
        )
        assert store.count() == 2  # no duplica, reemplaza

        # 5. Persistencia entre instancias (simula reinicio de proceso)
        store.close()
        store_reopened = VectorStore(path)
        assert store_reopened.count() == 2
        reopened_result = store_reopened.search(query_vector, n_results=1)[0]
        assert reopened_result["text"] == "Pérdida de circulación corregida y ampliada."

        # 6. Reinicio completo del índice (ej. cambio de modelo de embeddings)
        store_reopened.reset()
        assert store_reopened.count() == 0
        assert store_reopened.search(query_vector, n_results=5) == []

        # 7. Cierre explícito sin errores
        store_reopened.close()
