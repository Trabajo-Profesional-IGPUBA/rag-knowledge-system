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

    def test_ca11_initializes_without_external_server(self, tmp_path):
        """CA-1.1: La base vectorial debe iniciarse sin requerir Docker ni servidor externo."""
        store = VectorStore(tmp_path / "vs_ca11")
        assert store.count() == 0

    def test_ca11_creates_persist_directory(self, tmp_path):
        """CA-1.1: El directorio de persistencia debe crearse automáticamente."""
        path = tmp_path / "nuevo_directorio" / "vectorstore"
        VectorStore(path)
        assert path.exists()

    def test_ca12_data_persists_after_reinit(self, tmp_path):
        """CA-1.2: Los datos deben persistir en disco entre instancias del sistema."""
        path = tmp_path / "vs_persist"
        s1 = VectorStore(path)
        s1.add(
            "doc1::chunk_0",
            "pérdida de circulación en Quintuco",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        s2 = VectorStore(path)
        assert s2.count() == 1

    def test_ca12_search_works_after_reinit(self, tmp_path):
        """CA-1.2: La búsqueda debe funcionar correctamente tras reiniciar la instancia."""
        path = tmp_path / "vs_search_persist"
        s1 = VectorStore(path)
        s1.add(
            "doc1::chunk_0",
            "texto técnico",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        s2 = VectorStore(path)
        results = s2.search(_fake_embedding(), n_results=1)
        assert len(results) == 1

    def test_ca13_collection_created_on_first_init(self, tmp_path):
        """CA-1.3: La colección debe crearse si no existe previamente."""
        store = VectorStore(tmp_path / "vs_new")
        assert store.count() == 0

    def test_ca13_collection_recovered_on_second_init(self, tmp_path):
        """CA-1.3: La colección debe recuperarse si ya fue creada previamente."""
        path = tmp_path / "vs_recovery"
        s1 = VectorStore(path)
        s1.add(
            "doc1::chunk_0",
            "texto",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        s2 = VectorStore(path)
        assert s2.count() == 1

    def test_ca21_add_stores_text_and_metadata(self, store):
        """CA-2.1: La inserción debe almacenar el texto y la metadata del chunk."""
        store.add(
            "doc1::chunk_0",
            "workover en pozo LL-205",
            _fake_embedding(),
            {"doc_id": "doc1", "doc_type": "workover", "año": "2019"},
        )
        results = store.search(_fake_embedding(), n_results=1)
        assert results[0]["text"] == "workover en pozo LL-205"
        assert results[0]["metadata"]["doc_type"] == "workover"

    def test_ca21_batch_add_stores_all_chunks(self, store):
        """CA-2.1: La inserción batch debe almacenar todos los chunks correctamente."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc1::chunk_1"],
            texts=["texto chunk 0", "texto chunk 1"],
            embeddings=[_fake_embedding()] * 2,
            metadatas=[
                {"doc_id": "doc1", "doc_type": "ewrs"},
                {"doc_id": "doc1", "doc_type": "ewrs"},
            ],
        )
        assert store.count() == 2

    def test_ca22_upsert_same_id_does_not_duplicate(self, store):
        """CA-2.2: Reinsertar un chunk con el mismo ID no debe generar duplicados."""
        emb = _fake_embedding()
        store.add(
            "doc1::chunk_0", "versión 1", emb, {"doc_id": "doc1", "doc_type": "ewrs"}
        )
        store.add(
            "doc1::chunk_0", "versión 2", emb, {"doc_id": "doc1", "doc_type": "ewrs"}
        )
        assert store.count() == 1

    def test_ca22_upsert_updates_existing_content(self, store):
        """CA-2.2: Al reinsertar un chunk existente, el contenido debe actualizarse."""
        emb = _fake_embedding()
        store.add(
            "doc1::chunk_0",
            "texto original",
            emb,
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        store.add(
            "doc1::chunk_0",
            "texto actualizado",
            emb,
            {"doc_id": "doc1", "doc_type": "ewrs"},
        )
        results = store.search(emb, n_results=1)
        assert results[0]["text"] == "texto actualizado"

    def test_ca23_delete_removes_all_chunks_of_doc(self, store):
        """CA-2.3: Eliminar por doc_id debe remover todos los chunks de ese documento."""
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc1::chunk_1", "doc2::chunk_0"],
            texts=["chunk 0", "chunk 1", "chunk doc2"],
            embeddings=[_fake_embedding()] * 3,
            metadatas=[
                {"doc_id": "doc1", "doc_type": "ewrs"},
                {"doc_id": "doc1", "doc_type": "ewrs"},
                {"doc_id": "doc2", "doc_type": "parte_diario"},
            ],
        )
        store.delete_by_doc("doc1")
        assert store.count() == 1

    def test_ca23_delete_nonexistent_doc_does_not_raise(self, store):
        """CA-2.3: Eliminar un doc_id que no existe no debe lanzar excepción."""
        store.delete_by_doc("doc_inexistente")
        assert store.count() == 0
