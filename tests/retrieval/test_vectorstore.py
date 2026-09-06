"""
Tests para src/retrieval/vectorstore.py
Cubren "Persistencia de resultados":
  - Asociación entre chunk y embedding     -> CA-9.1, CA-9.2
  - Almacenamiento de embeddings generados -> CA-10.1 a CA-10.4
"""

import pytest

from src.retrieval.vectorstore import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "vectorstore")


def _fake_embedding(dim: int = 384) -> list[float]:
    return [0.1] * dim


class TestVectorStore:
    def test_initializes_empty(self, store):
        assert store.count() == 0

    def test_add_single_chunk(self, store):
        store.add(
            chunk_id="doc1::chunk_0",
            text="Pérdida de circulación en Quintuco.",
            embedding=_fake_embedding(),
            metadata={"doc_id": "doc1", "doc_type": "ewrs"},
        )
        assert store.count() == 1

    def test_add_batch(self, store):
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

    def test_add_batch_empty_does_nothing(self, store):
        store.add_batch([], [], [], [])
        assert store.count() == 0

    def test_upsert_does_not_duplicate(self, store):
        emb = _fake_embedding()
        store.add("doc1::chunk_0", "texto", emb, {"doc_id": "doc1", "doc_type": "ewrs"})
        store.add(
            "doc1::chunk_0",
            "texto actualizado",
            emb,
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        assert store.count() == 1

    def test_search_returns_results(self, store):
        store.add(
            "doc1::chunk_0",
            "circulación en Quintuco",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        results = store.search(_fake_embedding(), n_results=1)
        assert len(results) == 1

    def test_search_result_has_expected_keys(self, store):
        store.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        results = store.search(_fake_embedding(), n_results=1)
        assert "chunk_id" in results[0]
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]
        assert "distance" in results[0]

    def test_search_with_filter(self, store):
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

    def test_delete_by_doc(self, store):
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc2::chunk_0"],
            texts=["texto 1", "texto 2"],
            embeddings=[_fake_embedding()] * 2,
            metadatas=[
                {"doc_id": "doc1", "doc_type": "ewrs"},
                {"doc_id": "doc2", "doc_type": "ewrs"},
            ],
        )
        store.delete_by_doc("doc1")
        assert store.count() == 1

    def test_reset_empties_store(self, store):
        store.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        store.reset()
        assert store.count() == 0

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "vs"
        s1 = VectorStore(path)
        s1.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        s2 = VectorStore(path)
        assert s2.count() == 1

    # -----------------------------------------------------------------
    # Asociación entre chunk y embedding
    # -----------------------------------------------------------------

    def test_embedding_associated_to_fragment_by_position_before_storing(self, store):
        """CA-9.1: Antes de almacenar los resultados, cada embedding generado debe quedar asociado a su fragmento de origen (identificador, texto y metadata correspondientes), manteniendo esa correspondencia por posición."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc1::chunk_1"],
            texts=["texto A", "texto B"],
            embeddings=[[0.1] * 384, [0.9] * 384],
            metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc1"}],
        )
        result = store.search([0.9] * 384, n_results=1)[0]
        assert result["chunk_id"] == "doc1::chunk_1"
        assert result["text"] == "texto B"

    def test_stored_metadata_includes_minimum_required_fields(self, store):
        """CA-9.2: La metadata almacenada junto a cada fragmento debe incluir como mínimo el identificador del documento, su tipo, la posición del fragmento dentro del documento, la ruta de origen y el nombre de archivo."""
        store.add(
            chunk_id="doc1::chunk_0",
            text="texto",
            embedding=_fake_embedding(),
            metadata={
                "doc_id": "doc1",
                "doc_type": "ewrs",
                "chunk_index": 0,
                "source_path": "/raw/doc1.pdf",
                "filename": "doc1.pdf",
            },
        )
        result = store.search(_fake_embedding(), n_results=1)[0]
        for field in ("doc_id", "doc_type", "chunk_index", "source_path", "filename"):
            assert field in result["metadata"]

    # -----------------------------------------------------------------
    # Almacenamiento de embeddings generados
    # -----------------------------------------------------------------

    def test_can_index_single_fragment_with_text_embedding_and_metadata(self, store):
        """CA-10.1: El sistema debe permitir indexar un fragmento individual (con su texto, embedding y metadata) en la base vectorial."""
        store.add(
            chunk_id="doc1::chunk_0",
            text="Pérdida de circulación en Quintuco.",
            embedding=_fake_embedding(),
            metadata={"doc_id": "doc1", "doc_type": "ewrs"},
        )
        assert store.count() == 1

    def test_can_index_full_batch_in_single_operation(self, store):
        """CA-10.2: El sistema debe permitir indexar un lote completo de fragmentos en una sola operación, sin realizar ninguna acción cuando el lote está vacío."""
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

    def test_indexing_empty_batch_does_nothing(self, store):
        """CA-10.2: El sistema debe permitir indexar un lote completo de fragmentos en una sola operación, sin realizar ninguna acción cuando el lote está vacío."""
        store.add_batch([], [], [], [])
        assert store.count() == 0

    def test_storage_is_idempotent_reindexing_updates_not_duplicates(self, store):
        """CA-10.3: El almacenamiento debe ser idempotente: volver a indexar un fragmento ya existente debe actualizar su contenido, no crear un registro duplicado."""
        emb = _fake_embedding()
        store.add("doc1::chunk_0", "texto", emb, {"doc_id": "doc1", "doc_type": "ewrs"})
        store.add(
            "doc1::chunk_0",
            "texto actualizado",
            emb,
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        assert store.count() == 1
        assert store.search(emb, n_results=1)[0]["text"] == "texto actualizado"

    def test_stored_data_persists_across_separate_instances(self, tmp_path):
        """CA-10.4: Los datos almacenados deben persistir en disco y estar disponibles en corridas o procesos posteriores, sin necesidad de mantener viva la misma instancia del componente de almacenamiento entre ejecuciones."""
        path = tmp_path / "vs"
        s1 = VectorStore(path)
        s1.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        s2 = VectorStore(path)
        assert s2.count() == 1

    # -----------------------------------------------------------------
    # Validación de integridad de datos
    # -----------------------------------------------------------------

    def test_batch_indexing_validates_matching_lengths(self, store):
        """CA-11.1: Antes de indexar un lote, el sistema debe validar que la cantidad de identificadores, textos, embeddings y metadatas coincida entre sí, fallando de forma explícita si alguna de esas listas tiene una longitud distinta."""
        with pytest.raises(ValueError, match="misma longitud"):
            store.add_batch(
                chunk_ids=["doc1::chunk_0", "doc1::chunk_1"],
                texts=["solo un texto"],
                embeddings=[_fake_embedding(), _fake_embedding()],
                metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc1"}],
            )

    def test_batch_indexing_with_matching_lengths_still_works(self, store):
        """CA-11.1 (regresión): la validación agregada no debe romper el caso normal, donde chunk_ids, texts, embeddings y metadatas sí coinciden en longitud."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc1::chunk_1"],
            texts=["texto A", "texto B"],
            embeddings=[_fake_embedding(), _fake_embedding()],
            metadatas=[{"doc_id": "doc1"}, {"doc_id": "doc1"}],
        )
        assert store.count() == 2

    def test_search_result_includes_all_expected_fields_even_with_few_items(
        self, store
    ):
        """CA-11.2: Al buscar fragmentos relevantes, cada resultado devuelto debe incluir el identificador del fragmento, su texto, su metadata, la distancia calculada y el puntaje de similitud correspondiente, incluso cuando la base de datos tiene muy pocos elementos indexados."""
        store.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        result = store.search(_fake_embedding(), n_results=1)[0]
        for key in ("chunk_id", "text", "metadata", "distance", "score"):
            assert key in result

    def test_does_not_request_more_results_than_available_in_index(self, store):
        """CA-11.3: El sistema no debe solicitar más resultados de los que existen actualmente en el índice, para evitar fallas cuando la base de datos contiene pocos elementos."""
        store.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        results = store.search(_fake_embedding(), n_results=50)
        assert len(results) == 1

    def test_searching_empty_index_does_not_raise(self, store):
        """CA-11.3: El sistema no debe solicitar más resultados de los que existen actualmente en el índice, para evitar fallas cuando la base de datos contiene pocos elementos."""
        results = store.search(_fake_embedding(), n_results=5)
        assert results == []
