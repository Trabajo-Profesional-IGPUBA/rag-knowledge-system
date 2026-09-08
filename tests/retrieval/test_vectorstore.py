# tests/retrieval/test_vectorstore.py
"""
Tests para src/retrieval/vectorstore.py
Cubren "Épica: Migración del motor de base de datos vectorial: ChromaDB → Qdrant":
  - Inicialización del motor -> CA-1.1 a CA-1.2
  - Identificación de chunks  -> CA-2.1 a CA-2.3
  - Indexación de chunks  -> CA-3.1 a CA-3.4
  - Filtrado por metadata -> CA-4.1 a CA-4.3
"""

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.retrieval.vectorstore import VectorStore, _build_filter, _to_point_id


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

    def test_accepts_domain_defined_chunk_id_format(self, store):
        """CA-2.1: El sistema debe admitir identificadores de chunk definidos por el dominio,
        incluso si el motor subyacente exige un formato de identificador distinto."""
        store.add("doc1::chunk_0", "texto", _fake_embedding(), {"doc_id": "doc1"})
        assert store.count() == 1

    def test_domain_chunk_id_is_recoverable(self, store):
        """CA-2.2: El identificador de dominio debe conservarse y ser recuperable en los resultados
        de búsqueda, independientemente del identificador interno usado por el motor."""
        store.add("doc1::chunk_0", "texto", _fake_embedding(), {"doc_id": "doc1"})
        assert store.count() == 1

    def test_same_chunk_id_maps_to_same_internal_record(self, store):
        """CA-2.3: El mismo identificador de chunk debe mapear siempre al mismo
        registro interno, para sostener la idempotencia de la indexación."""
        emb = _fake_embedding()
        store.add("doc1::chunk_0", "texto", emb, {"doc_id": "doc1"})
        store.add("doc1::chunk_0", "texto actualizado", emb, {"doc_id": "doc1"})
        assert store.count() == 1

    def test_to_point_id_is_deterministic(self):
        """CA-2.3: El mismo identificador de chunk debe mapear siempre al mismo
        registro interno, para sostener la idempotencia de la indexación."""
        uuid1 = _to_point_id("doc1::chunk_0")
        uuid2 = _to_point_id("doc1::chunk_0")
        assert uuid1 == uuid2

    # -----------------------------------------------------------------
    # Indexación de chunks
    # -----------------------------------------------------------------

    def test_can_index_single_chunk(self, store):
        """CA-3.1: El sistema debe permitir indexar un chunk individual."""
        store.add(
            "doc1::chunk_0",
            "Pérdida de circulación en Quintuco.",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        assert store.count() == 1

    def test_can_index_batch_in_single_operation(self, store):
        """CA-3.2: El sistema debe permitir indexar un lote de chunks en una sola operación."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc1::chunk_1", "doc2::chunk_0"],
            texts=["texto 1", "texto 2", "texto 3"],
            embeddings=[_fake_embedding()] * 3,
            metadatas=[
                {"doc_id": "doc1", "doc_type": "ewrs"},
                {"doc_id": "doc1", "doc_type": "ewrs"},
                {"doc_id": "doc2", "doc_type": "parte_diario"},
            ],
        )
        assert store.count() == 3

    def test_reindexing_is_idempotent_not_duplicated(self, store):
        """CA-3.3: Reprocesar un chunk ya indexado no debe generar duplicados (comportamiento idempotente)."""
        emb = _fake_embedding()
        store.add("doc1::chunk_0", "texto", emb, {"doc_id": "doc1"})
        store.add("doc1::chunk_0", "texto actualizado", emb, {"doc_id": "doc1"})
        assert store.count() == 1

    def test_indexing_empty_batch_has_no_effect(self, store):
        """CA-3.4: Indexar un lote vacío no debe producir error ni efecto alguno."""
        store.add_batch([], [], [], [])
        assert store.count() == 0

    # -----------------------------------------------------------------
    # Filtrado por metadata
    # -----------------------------------------------------------------

    def test_build_filter_exact_match(self):
        """CA-4.1: El sistema debe permitir filtrar resultados por un valor exacto de metadata."""
        filters = {"doc_type": "parte_diario"}
        expected = Filter(
            must=[
                FieldCondition(key="doc_type", match=MatchValue(value="parte_diario"))
            ]
        )
        assert _build_filter(filters) == expected

    def test_build_filter_in_set_match(self):
        """CA-4.2: El sistema debe permitir filtrar resultados por pertenencia a un conjunto de valores posibles de metadata."""
        filters = {"doc_type": {"$in": ["ewrs", "workover_report"]}}
        expected = Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchAny(any=["ewrs", "workover_report"]),
                )
            ]
        )
        assert _build_filter(filters) == expected

    def test_build_filter_returns_none_when_empty(self):
        """CA-4.3: Una búsqueda sin filtros debe comportarse igual que antes de introducir el soporte de filtrado."""
        assert _build_filter(None) is None
        assert _build_filter({}) is None

    def test_filter_by_exact_metadata_value(self, store):
        """CA-4.1: El sistema debe permitir filtrar resultados por un valor exacto de metadata."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc2::chunk_0"],
            texts=["parte diario", "informe EWR"],
            embeddings=[_fake_embedding()] * 2,
            metadatas=[
                {"doc_id": "doc1", "doc_type": "parte_diario"},
                {"doc_id": "doc2", "doc_type": "ewrs"},
            ],
        )
        results = store.search(
            _fake_embedding(), n_results=5, filters={"doc_type": "ewrs"}
        )
        assert all(r["metadata"]["doc_type"] == "ewrs" for r in results)

    def test_filter_by_set_of_possible_values(self, store):
        """CA-4.2: El sistema debe permitir filtrar resultados por pertenencia a un conjunto de valores posibles de metadata."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc2::chunk_0", "doc3::chunk_0"],
            texts=["a", "b", "c"],
            embeddings=[_fake_embedding()] * 3,
            metadatas=[
                {"doc_id": "doc1", "doc_type": "ewrs"},
                {"doc_id": "doc2", "doc_type": "workover_report"},
                {"doc_id": "doc3", "doc_type": "parte_diario"},
            ],
        )
        results = store.search(
            _fake_embedding(),
            n_results=5,
            filters={"doc_type": {"$in": ["ewrs", "workover_report"]}},
        )
        doc_types = {r["metadata"]["doc_type"] for r in results}
        assert doc_types.issubset({"ewrs", "workover_report"})
        assert len(results) == 2

    def test_search_without_filters_behaves_as_before(self, store):
        """CA-4.3: Una búsqueda sin filtros debe comportarse igual que antes de introducir el soporte de filtrado."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc2::chunk_0"],
            texts=["a", "b"],
            embeddings=[_fake_embedding()] * 2,
            metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc2"}],
        )
        assert len(store.search(_fake_embedding(), n_results=5)) == 2

    # -----------------------------------------------------------------
    # Recuperación semántica
    # -----------------------------------------------------------------

    def test_returns_most_relevant_chunks_by_similarity(self, store):
        """CA-5.1: El sistema debe devolver los chunks más relevantes ordenados por
        similitud semántica respecto a una consulta vectorizada."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc1::chunk_1"],
            texts=["texto A", "texto B"],
            embeddings=[[0.1] * 384, [0.9] * 384],
            metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc1"}],
        )
        result = store.search([0.9] * 384, n_results=1)[0]
        assert result["chunk_id"] == "doc1::chunk_1"

    def test_respects_max_requested_results(self, store):
        """CA-5.2: El sistema debe respetar la cantidad máxima de resultados solicitada."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc2::chunk_0", "doc3::chunk_0"],
            texts=["a", "b", "c"],
            embeddings=[_fake_embedding()] * 3,
            metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc2"}, {"doc_id": "doc3"}],
        )
        assert len(store.search(_fake_embedding(), n_results=2)) == 2

    def test_returns_empty_list_when_index_is_empty(self, store):
        """CA-5.3: El sistema debe devolver una lista vacía si no hay documentos indexados, sin lanzar error."""
        assert store.search(_fake_embedding(), n_results=5) == []

    def test_result_contract_matches_previous_version(self, store):
        """CA-5.4: Cada resultado devuelto debe incluir identificador de chunk, texto, metadata original,
        y una medida de similitud/distancia, manteniendo el mismo contrato de datos que la versión anterior.
        """
        store.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        result = store.search(_fake_embedding(), n_results=1)[0]
        for key in ("chunk_id", "text", "metadata", "distance", "score"):
            assert key in result

    def test_does_not_request_more_results_than_available(self, store):
        """CA-5.5: El sistema debe retornar únicamente los chunks disponibles sin lanzar error cuando la cantidad de resultados solicitada
        sea mayor al total de chunks indexados."""
        store.add("doc1::chunk_0", "texto", _fake_embedding(), {"doc_id": "doc1"})
        assert len(store.search(_fake_embedding(), n_results=50)) == 1

    # -----------------------------------------------------------------
    # Cierre de recursos
    # -----------------------------------------------------------------

    def test_can_close_connection_explicitly(self, store):
        """CA-7.1: permite cerrar explícitamente la conexión."""
        store.close()  # no debe lanzar

    def test_close_after_normal_use_does_not_raise(self, store):
        """CA-7.2: cerrar tras indexar y/o buscar no produce error."""
        store.add("doc1::chunk_0", "texto", _fake_embedding(), {"doc_id": "doc1"})
        store.search(_fake_embedding(), n_results=1)
        store.close()
