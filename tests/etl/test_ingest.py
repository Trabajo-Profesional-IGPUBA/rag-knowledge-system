import pytest

from ingest_files import DEFAULT_MAX_WORKERS, existing_dir, validate_max_workers

"""
Cubren "Paralelizar la ingesta de documentos (ETL)":
  - Configuración de la concurrencia-> CA-1.1 a CA-1.6
  - Procesamiento de archivos-> CA-2.1 a CA-2.3
  - Integridad y compatibilidad-> CA-3.1 a CA-3.2
  - Observabilidad de la ingesta-> CA-4.1 a CA-4.2
  - Liberación de recursos-> CA-5.1 a CA-5.2
"""


def make_fake_processor(processed_files=None, fail_on=None):
    class _FakeProcessor:
        def process_file(self, file):
            if fail_on and file.name in fail_on:
                raise RuntimeError("fallo simulado")
            if processed_files is not None:
                processed_files.append(file.name)
            return type("M", (), {"n_chunks": 1, "time_total_s": 0.01})()

    return _FakeProcessor()


def patch_dependencies(monkeypatch, processor, close_calls=None):
    fake_store = type(
        "FakeStore",
        (),
        {
            "close": lambda self: (
                close_calls.append(True) if close_calls is not None else None
            )
        },
    )()
    monkeypatch.setattr(
        "src.retrieval.vectorstore.VectorStore", lambda *a, **k: fake_store
    )
    monkeypatch.setattr("src.embeddings.embedder.Embedder", lambda: object())
    monkeypatch.setattr("src.etl.DoclingHybridChunker", lambda: object())
    monkeypatch.setattr("src.etl.DocumentProcessor", lambda **kwargs: processor)
    return fake_store


class TestConfigurableConcurrency:

    def test_allows_configuring_number_of_concurrent_files(self):
        """CA-1.1: El sistema debe permitir configurar cuántos archivos se
        procesan al mismo tiempo."""
        assert validate_max_workers("3") == 3

    def test_missing_config_uses_reasonable_default(self, monkeypatch):
        """CA-1.2: Si no se proporciona ninguna configuración, el sistema
        debe continuar funcionando utilizando un valor por defecto razonable."""
        monkeypatch.delenv("INGEST_MAX_WORKERS", raising=False)
        assert DEFAULT_MAX_WORKERS >= 1

    def test_non_numeric_config_raises(self):
        """CA-1.3: Si se proporciona una configuración cuyo formato no es
        válido, el sistema debe rechazarla."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            validate_max_workers("abc")

    def test_explicit_zero_workers_is_rejected(self):
        """CA-1.5: Si se solicita explícitamente un valor de concurrencia
        imposible al iniciar el proceso, el sistema debe rechazarlo."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            validate_max_workers("0")

    def test_explicit_negative_workers_is_rejected(self):
        """CA-1.5: ídem, con un valor negativo."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            validate_max_workers("-5")

    def test_never_submits_more_concurrent_tasks_than_available_files(
        self, tmp_path, monkeypatch
    ):
        """CA-1.6: El sistema no debe reservar más capacidad de procesamiento
        simultáneo que la cantidad de archivos disponibles para procesar."""
        import threading

        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf"]:
            (raw_dir / name).touch()

        max_concurrent_seen = 0
        currently_running = 0
        lock = threading.Lock()

        class _FakeProcessor:
            def process_file(self, file):
                nonlocal max_concurrent_seen, currently_running
                with lock:
                    currently_running += 1
                    max_concurrent_seen = max(max_concurrent_seen, currently_running)
                import time

                time.sleep(0.05)
                with lock:
                    currently_running -= 1
                return type("M", (), {"n_chunks": 1, "time_total_s": 0.01})()

        patch_dependencies(monkeypatch, _FakeProcessor())
        run_module.run(sources=[raw_dir], max_workers=10)

        assert max_concurrent_seen <= 2


class TestArchivesProcessing:

    def test_processes_multiple_files(self, tmp_path, monkeypatch):
        """CA-2.1: El sistema debe poder procesar varios archivos al mismo tiempo."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (raw_dir / name).touch()

        processed_files = []
        patch_dependencies(
            monkeypatch, make_fake_processor(processed_files=processed_files)
        )
        run_module.run(sources=[raw_dir], max_workers=2)

        assert sorted(processed_files) == ["a.pdf", "b.pdf", "c.pdf"]

    def test_one_file_failing_does_not_stop_the_rest(self, tmp_path, monkeypatch):
        """CA-2.2: Si un archivo falla durante el procesamiento, el resto
        debe seguir procesándose con normalidad."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (raw_dir / name).touch()

        processed_files = []
        patch_dependencies(
            monkeypatch,
            make_fake_processor(processed_files=processed_files, fail_on={"b.pdf"}),
        )
        run_module.run(sources=[raw_dir], max_workers=2)

        assert sorted(processed_files) == ["a.pdf", "c.pdf"]

    def test_no_files_is_reported_and_finishes_without_error(
        self, tmp_path, monkeypatch, caplog
    ):
        """CA-2.3: Si no hay archivos para procesar, el sistema debe
        informarlo y finalizar sin error."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        patch_dependencies(monkeypatch, make_fake_processor())

        with caplog.at_level("WARNING"):
            run_module.run(sources=[raw_dir], max_workers=2)

        assert "No se encontraron archivos PDF" in caplog.text


class TestSources:

    def test_single_pdf_file_as_source(self, tmp_path, monkeypatch):
        """Passing a single PDF file as source processes only that file."""
        import ingest_files as run_module

        pdf = tmp_path / "solo.pdf"
        pdf.touch()

        processed_files = []
        patch_dependencies(
            monkeypatch, make_fake_processor(processed_files=processed_files)
        )
        run_module.run(sources=[pdf], max_workers=1)

        assert processed_files == ["solo.pdf"]

    def test_multiple_directories_as_sources(self, tmp_path, monkeypatch):
        """Passing multiple directories processes PDFs from all of them."""
        import ingest_files as run_module

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "x.pdf").touch()
        (dir_b / "y.pdf").touch()

        processed_files = []
        patch_dependencies(
            monkeypatch, make_fake_processor(processed_files=processed_files)
        )
        run_module.run(sources=[dir_a, dir_b], max_workers=2)

        assert sorted(processed_files) == ["x.pdf", "y.pdf"]

    def test_mix_of_file_and_directory_as_sources(self, tmp_path, monkeypatch):
        """Passing a mix of a file and a directory processes both."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "a.pdf").touch()
        single_pdf = tmp_path / "b.pdf"
        single_pdf.touch()

        processed_files = []
        patch_dependencies(
            monkeypatch, make_fake_processor(processed_files=processed_files)
        )
        run_module.run(sources=[raw_dir, single_pdf], max_workers=2)

        assert sorted(processed_files) == ["a.pdf", "b.pdf"]

    def test_discovers_pdfs_recursively_in_directory(self, tmp_path, monkeypatch):
        """PDFs in subdirectories are discovered recursively."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        subdir = raw_dir / "ewr"
        subdir.mkdir(parents=True)
        (subdir / "deep.pdf").touch()

        processed_files = []
        patch_dependencies(
            monkeypatch, make_fake_processor(processed_files=processed_files)
        )
        run_module.run(sources=[raw_dir], max_workers=1)

        assert processed_files == ["deep.pdf"]

    def test_non_pdf_files_in_directory_are_ignored(self, tmp_path, monkeypatch):
        """Non-PDF files in a source directory are not processed."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc.pdf").touch()
        (raw_dir / "readme.txt").touch()
        (raw_dir / "data.csv").touch()

        processed_files = []
        patch_dependencies(
            monkeypatch, make_fake_processor(processed_files=processed_files)
        )
        run_module.run(sources=[raw_dir], max_workers=1)

        assert processed_files == ["doc.pdf"]

    def test_existing_dir_accepts_pdf_file(self, tmp_path):
        """existing_dir accepts a valid PDF file path."""
        pdf = tmp_path / "valid.pdf"
        pdf.touch()
        assert existing_dir(str(pdf)) == pdf

    def test_existing_dir_accepts_directory(self, tmp_path):
        """existing_dir accepts a valid directory path."""
        assert existing_dir(str(tmp_path)) == tmp_path

    def test_existing_dir_rejects_missing_path(self, tmp_path):
        """existing_dir raises for a path that does not exist."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            existing_dir(str(tmp_path / "nonexistent"))

    def test_existing_dir_rejects_non_pdf_file(self, tmp_path):
        """existing_dir raises for a file that is not a PDF."""
        import argparse

        txt = tmp_path / "notes.txt"
        txt.touch()
        with pytest.raises(argparse.ArgumentTypeError):
            existing_dir(str(txt))


class TestConcurrentIndexingIntegrityAndCompatibility:

    def test_concurrent_indexing_never_corrupts_or_loses_data(self):
        """CA-3.1: Procesar archivos de forma concurrente nunca debe
        corromper, perder ni duplicar los datos ya almacenados."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from ingest_files import _SerializedVectorStore

        call_log: list[str] = []
        in_critical_section = threading.Event()

        class _FakeInnerStore:
            def add(self, *a, **k):
                assert not in_critical_section.is_set()
                in_critical_section.set()
                call_log.append("add")
                in_critical_section.clear()

            def add_batch(self, *a, **k):
                assert not in_critical_section.is_set()
                in_critical_section.set()
                call_log.append("add_batch")
                in_critical_section.clear()

        lock = threading.Lock()
        safe_store = _SerializedVectorStore(_FakeInnerStore(), lock)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(20):
                if i % 2 == 0:
                    futures.append(
                        executor.submit(safe_store.add, f"id{i}", "t", [], {})
                    )
                else:
                    futures.append(
                        executor.submit(safe_store.add_batch, [], [], [], [])
                    )
            for f in futures:
                f.result()

        assert len(call_log) == 20

    def test_reprocessing_same_content_concurrently_does_not_duplicate(self, tmp_path):
        """CA-3.1: ídem, verificado contra el VectorStore real."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from ingest_files import _SerializedVectorStore
        from src.retrieval.vectorstore import VectorStore

        real_store = VectorStore(tmp_path / "vs", embedding_dim=4)
        lock = threading.Lock()
        safe_store = _SerializedVectorStore(real_store, lock)
        emb = [0.1, 0.2, 0.3, 0.4]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    safe_store.add,
                    "doc1::chunk_0",
                    f"texto v{i}",
                    emb,
                    {"doc_id": "doc1"},
                )
                for i in range(10)
            ]
            for f in futures:
                f.result()

        assert real_store.count() == 1
        real_store.close()

    def test_internal_tokenizer_parallelism_is_disabled_on_import(self, monkeypatch):
        """CA-3.2: El procesamiento concurrente no debe generar conflictos
        con otras herramientas internas que también manejan su propio
        paralelismo."""
        import importlib
        import os

        import ingest_files as run_module

        monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
        importlib.reload(run_module)

        assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"


class TestObservability:

    def test_reports_success_and_failure_counts_at_the_end(
        self, tmp_path, monkeypatch, caplog
    ):
        """CA-4.1: Al finalizar la ingesta, el sistema debe informar
        cuántos archivos se procesaron con éxito y cuántos fallaron."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf"]:
            (raw_dir / name).touch()

        patch_dependencies(
            monkeypatch,
            make_fake_processor(fail_on={"a.pdf"}),
        )

        with caplog.at_level("INFO"):
            run_module.run(sources=[raw_dir], max_workers=2)

        assert "OK=1" in caplog.text
        assert "Fallidos=1" in caplog.text

    def test_reports_periodic_progress_on_long_runs(
        self, tmp_path, monkeypatch, caplog
    ):
        """CA-4.2: En ejecuciones largas, el sistema debe informar el
        progreso periódicamente, no solo al final."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for i in range(120):
            (raw_dir / f"file_{i}.pdf").touch()

        patch_dependencies(monkeypatch, make_fake_processor())

        with caplog.at_level("INFO"):
            run_module.run(sources=[raw_dir], max_workers=4)

        progress_lines = [r for r in caplog.records if "Progreso:" in r.message]
        assert len(progress_lines) >= 1


class TestResourceCleanup:

    def test_storage_resources_released_even_if_files_fail(self, tmp_path, monkeypatch):
        """CA-5.1: Los recursos de almacenamiento utilizados deben
        liberarse siempre al finalizar la ingesta, incluso si hubo errores."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf"]:
            (raw_dir / name).touch()

        close_calls = []
        patch_dependencies(
            monkeypatch,
            make_fake_processor(fail_on={"a.pdf"}),
            close_calls=close_calls,
        )
        run_module.run(sources=[raw_dir], max_workers=2)

        assert close_calls == [True]

    def test_no_resources_opened_if_no_files_found(self, tmp_path, monkeypatch):
        """CA-5.2: ídem, cuando no hay archivos para procesar."""
        import ingest_files as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        store_created = []
        monkeypatch.setattr(
            "src.retrieval.vectorstore.VectorStore",
            lambda *a, **k: store_created.append(True),
        )
        run_module.run(sources=[raw_dir], max_workers=2)

        assert store_created == []
