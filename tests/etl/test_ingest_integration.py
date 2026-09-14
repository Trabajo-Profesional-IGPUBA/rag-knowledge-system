"""
Test de integración para la ingesta (main.py).

Corre run() de punta a punta con componentes reales, no con mocks.
"""

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pdfs"

PVT_FIXTURE = FIXTURES_DIR / "pvt_los_perales.pdf"
INY_FIXTURE = FIXTURES_DIR / "iny_pi08.pdf"


@pytest.mark.integration
class TestIngestEndToEnd:

    def test_full_pipeline_indexes_real_pdf_with_real_components(
        self, tmp_path, monkeypatch
    ):
        import main as run_module
        from src.retrieval.vectorstore import VectorStore

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"

        assert PVT_FIXTURE.exists(), f"Falta el fixture {PVT_FIXTURE}."
        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")

        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        run_module.run(max_workers=2)

        store = VectorStore(vector_store_dir)
        try:
            assert store.count() >= 1
        finally:
            store.close()

    def test_reingesting_same_pdf_is_idempotent_end_to_end(self, tmp_path, monkeypatch):
        import main as run_module
        from src.retrieval.vectorstore import VectorStore

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"

        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")

        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        run_module.run(max_workers=2)
        store = VectorStore(vector_store_dir)
        count_after_first_run = store.count()
        store.close()

        run_module.run(max_workers=2)

        store = VectorStore(vector_store_dir)
        try:
            assert store.count() == count_after_first_run
        finally:
            store.close()

    def test_processes_multiple_real_files_concurrently_without_corruption(
        self, tmp_path, monkeypatch, caplog
    ):
        """CA-3.1/3.2: 6+ PDFs reales y distintos, max_workers=2
        (max_in_flight=4), para forzar la reposición del `while` y no
        solo el envío inicial."""
        import main as run_module
        from src.retrieval.vectorstore import VectorStore

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"

        source_files = sorted(FIXTURES_DIR.glob("*.pdf"))
        assert len(source_files) >= 6, (
            f"Se esperaban al menos 6 PDFs distintos en {FIXTURES_DIR}, "
            f"se encontraron {len(source_files)}."
        )

        for source_file in source_files:
            shutil.copy(source_file, raw_dir / source_file.name)

        total_files = len(source_files)
        max_workers = 2
        max_in_flight = max_workers * 2
        assert total_files > max_in_flight, (
            "Se necesitan más archivos que max_in_flight para ejercitar "
            "la reposición del while."
        )

        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        with caplog.at_level("INFO"):
            run_module.run(max_workers=max_workers)

        assert (
            f"Procesados={total_files} | OK={total_files} | Fallidos=0" in caplog.text
        )

        store = VectorStore(vector_store_dir)
        try:
            assert store.count() >= total_files
        finally:
            store.close()

    def test_real_processing_failure_does_not_abort_the_rest(
        self, tmp_path, monkeypatch
    ):
        """CA-2.2 + CA-5.1: un archivo corrupto no debe abortar el resto
        ni dejar el VectorStore en mal estado."""
        import main as run_module
        from src.retrieval.vectorstore import VectorStore

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"

        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        shutil.copy(INY_FIXTURE, raw_dir / "iny.pdf")
        (raw_dir / "corrupto.pdf").write_bytes(b"esto no es un PDF valido")

        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        run_module.run(max_workers=2)

        store = VectorStore(vector_store_dir)
        try:
            assert store.count() >= 2
        finally:
            store.close()

    def test_search_retrieves_the_correct_document_not_a_mix_of_both(
        self, tmp_path, monkeypatch
    ):
        """Detecta metadata mal mapeada entre hilos: cada búsqueda debe
        traer contenido del documento correcto, no una mezcla."""
        import main as run_module
        from src.embeddings.embedder import Embedder
        from src.retrieval.vectorstore import VectorStore

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        vector_store_dir = tmp_path / "vectorstore"

        shutil.copy(PVT_FIXTURE, raw_dir / "pvt.pdf")
        shutil.copy(INY_FIXTURE, raw_dir / "iny.pdf")

        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(run_module, "VECTOR_STORE_PATH", vector_store_dir)

        run_module.run(max_workers=2)

        store = VectorStore(vector_store_dir)
        embedder = Embedder()
        try:
            pvt_query = embedder.embed("presión de burbuja del yacimiento")
            pvt_results = store.search(pvt_query, n_results=1)
            assert len(pvt_results) == 1
            assert (
                "burbuja" in pvt_results[0]["text"].lower()
                or "pvt" in pvt_results[0]["text"].lower()
            )

            iny_query = embedder.embed("inyección continua de polímeros HPAM")
            iny_results = store.search(iny_query, n_results=1)
            assert len(iny_results) == 1
            assert (
                "polímero" in iny_results[0]["text"].lower()
                or "hpam" in iny_results[0]["text"].lower()
            )

            assert pvt_results[0]["chunk_id"] != iny_results[0]["chunk_id"]
        finally:
            store.close()
