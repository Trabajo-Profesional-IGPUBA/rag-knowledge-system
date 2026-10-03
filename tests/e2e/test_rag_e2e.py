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

    def test_pipeline_indexes_at_least_one_chunk(self, tmp_path):
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"

        assert PVT_FIXTURE.exists(), f"Falta el fixture {PVT_FIXTURE}."
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")

        run_module.run(sources=[raw_dir], max_workers=1)

        store = VectorStore(vector_store_dir)
        try:
            assert store.count() >= 1
        finally:
            store.close()