import shutil
from pathlib import Path

import pytest

from src.embeddings.embedder import Embedder
from src.retrieval.vectorstore import VectorStore
from src.retrieval.retriever import Retriever

FIXTURES_DIR = Path(__file__).parent.parent / "etl" / "fixtures" / "pdfs"
PVT_FIXTURE = FIXTURES_DIR / "pvt_los_perales.pdf"


@pytest.mark.e2e
class TestRagEndToEnd:

    def test_pipeline_indexes_at_least_one_chunk(self, tmp_path, monkeypatch):
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"

        assert PVT_FIXTURE.exists(), f"Falta el fixture {PVT_FIXTURE}."
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        run_module.run(sources=[raw_dir], max_workers=1)

        store = VectorStore(vector_store_dir)
        try:
            assert store.count() >= 1
        finally:
            store.close()

    def test_semantic_query_returns_at_least_one_result(self, tmp_path, monkeypatch):
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)
        run_module.run(sources=[raw_dir], max_workers=1)

        embedder = Embedder()
        store = VectorStore(vector_store_dir)
        retriever = Retriever(embedder=embedder, vectorstore=store)

        try:
            result = retriever.retrieve("presión de burbuja", top_k=3)
            assert len(result.chunks) >= 1
        finally:
            store.close()

    def test_retrieved_chunk_has_positive_score(self, tmp_path, monkeypatch):
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)
        run_module.run(sources=[raw_dir], max_workers=1)

        embedder = Embedder()
        store = VectorStore(vector_store_dir)
        retriever = Retriever(embedder=embedder, vectorstore=store)

        try:
            result = retriever.retrieve("presión de burbuja", top_k=3)
            assert result.best_score > 0
        finally:
            store.close()

    def test_domain_query_retrieves_relevant_chunks(self, tmp_path, monkeypatch):
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)
        run_module.run(sources=[raw_dir], max_workers=1)

        embedder = Embedder()
        store = VectorStore(vector_store_dir)
        retriever = Retriever(embedder=embedder, vectorstore=store)

        try:
            result = retriever.retrieve("presión de burbuja yacimiento", top_k=3)
            texts = " ".join(c["text"] for c in result.chunks).lower()
            assert any(term in texts for term in ["presión", "burbuja", "yacimiento", "psi"])
        finally:
            store.close()

    def test_full_pipeline_runs_without_errors(self, tmp_path, monkeypatch):
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        try:
            run_module.run(sources=[raw_dir], max_workers=1)
        except Exception as e:
            pytest.fail(f"El pipeline lanzó una excepción inesperada: {e}")

    def test_reingesting_same_pdf_does_not_duplicate_chunks(self, tmp_path, monkeypatch):
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        run_module.run(sources=[raw_dir], max_workers=1)
        store = VectorStore(vector_store_dir)
        count_first = store.count()
        store.close()

        run_module.run(sources=[raw_dir], max_workers=1)
        store = VectorStore(vector_store_dir)
        count_second = store.count()
        store.close()

        assert count_first == count_second