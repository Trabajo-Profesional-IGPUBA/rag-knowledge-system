"""
Test de integración para src/retrieval/vectorstore.py

A diferencia de test_vectorstore.py, este archivo ejercita el flujo real end-to-end contra Qdrant embebido,
sin mockear nada: simula el ciclo de vida completo tal como lo usaría el pipeline de indexación y recuperación en producción.
"""

from src.retrieval.vectorstore import VectorStore


def _fake_embedding(seed: float, dim: int = 384) -> list[float]:
    return [seed] * dim


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

        # 2. Indexación de un lote mixto
        store.add_batch(
            chunk_ids=["ewr_001::chunk_0", "ewr_001::chunk_1", "parte_002::chunk_0"],
            texts=[
                "Pérdida de circulación en Quintuco a 2500m.",
                "Se detecta gas en superficie, monitoreo continuo.",
                "Parte diario: turno noche sin novedades.",
            ],
            embeddings=[
                _fake_embedding(0.9),
                _fake_embedding(0.85),
                _fake_embedding(0.1),
            ],
            metadatas=[
                {"doc_id": "ewr_001", "doc_type": "ewrs", "chunk_index": 0},
                {"doc_id": "ewr_001", "doc_type": "ewrs", "chunk_index": 1},
                {"doc_id": "parte_002", "doc_type": "parte_diario", "chunk_index": 0},
            ],
        )
        assert store.count() == 3

        # 3. Búsqueda semántica con filtro
        results = store.search(
            _fake_embedding(0.9), n_results=5, filters={"doc_type": "ewrs"}
        )
        assert len(results) == 2
        assert results[0]["chunk_id"] == "ewr_001::chunk_0"  # más cercano al query
        for key in ("chunk_id", "text", "metadata", "distance", "score"):
            assert key in results[0]
        assert all(r["metadata"]["doc_type"] == "ewrs" for r in results)

        # 4. Reprocesamiento de un documento (simula corrección de contenido)
        store.delete_by_doc("ewr_001")
        assert store.count() == 1  # solo queda parte_002

        store.add(
            "ewr_001::chunk_0",
            "Pérdida de circulación corregida y ampliada.",
            _fake_embedding(0.9),
            {"doc_id": "ewr_001", "doc_type": "ewrs", "chunk_index": 0},
        )
        assert store.count() == 2  # no duplica, reemplaza

        # 5. Persistencia entre instancias (simula reinicio de proceso)
        store.close()
        store_reopened = VectorStore(path)
        assert store_reopened.count() == 2
        reopened_result = store_reopened.search(_fake_embedding(0.9), n_results=1)[0]
        assert reopened_result["text"] == "Pérdida de circulación corregida y ampliada."

        # 6. Reinicio completo del índice (ej. cambio de modelo de embeddings)
        store_reopened.reset()
        assert store_reopened.count() == 0
        assert store_reopened.search(_fake_embedding(0.9), n_results=5) == []

        # 7. Cierre explícito sin errores
        store_reopened.close()
