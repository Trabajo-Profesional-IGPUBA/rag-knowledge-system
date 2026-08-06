import pytest
from src.retrieval.vectorstore import VectorStore, COLLECTION_NAME


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
        store.add("doc1::chunk_0", "texto actualizado", emb, {"doc_id": "doc1", "doc_type": "ewrs"})
        assert store.count() == 1

    def test_search_returns_results(self, store):
        store.add("doc1::chunk_0", "circulación en Quintuco", _fake_embedding(), {"doc_id": "doc1", "doc_type": "ewrs"})
        results = store.search(_fake_embedding(), n_results=1)
        assert len(results) == 1

    def test_search_result_has_expected_keys(self, store):
        store.add("doc1::chunk_0", "texto", _fake_embedding(), {"doc_id": "doc1", "doc_type": "ewrs"})
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
        results = store.search(_fake_embedding(), n_results=5, filters={"doc_type": "ewrs"})
        assert all(r["metadata"]["doc_type"] == "ewrs" for r in results)

    def test_delete_by_doc(self, store):
        store.add_batch(
            chunk_ids=["doc1::chunk_0", "doc2::chunk_0"],
            texts=["texto 1", "texto 2"],
            embeddings=[_fake_embedding()] * 2,
            metadatas=[{"doc_id": "doc1", "doc_type": "ewrs"}, {"doc_id": "doc2", "doc_type": "ewrs"}],
        )
        store.delete_by_doc("doc1")
        assert store.count() == 1

    def test_reset_empties_store(self, store):
        store.add("doc1::chunk_0", "texto", _fake_embedding(), {"doc_id": "doc1", "doc_type": "ewrs"})
        store.reset()
        assert store.count() == 0

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "vs"
        s1 = VectorStore(path)
        s1.add("doc1::chunk_0", "texto", _fake_embedding(), {"doc_id": "doc1", "doc_type": "ewrs"})
        s2 = VectorStore(path)
        assert s2.count() == 1
