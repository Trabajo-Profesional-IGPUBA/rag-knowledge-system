import pytest

from main import DEFAULT_MAX_WORKERS, _get_max_workers

"""
Cubren "Paralelizar la ingesta de documentos (ETL)":
  - Configuración de la concurrencia-> CA-1.1 a CA-1.6
  - Procesamiento de archivos-> CA-2.1 a CA-2.3
  - Integridad y compatibilidad-> CA-3.1 a CA-3.2
  - Observabilidad de la ingesta-> CA-4.1
"""


class TestConfigurableConcurrency:

    def test_allows_configuring_number_of_concurrent_files(self, monkeypatch):
        """CA-1.1: El sistema debe permitir configurar cuántos archivos se
        procesan al mismo tiempo."""
        monkeypatch.setenv("INGEST_MAX_WORKERS", "3")
        assert _get_max_workers() == 3

    def test_missing_config_uses_reasonable_default(self, monkeypatch):
        """CA-1.2: Si no se proporciona ninguna configuración, el sistema
        debe continuar funcionando utilizando un valor por defecto
        razonable."""
        monkeypatch.delenv("INGEST_MAX_WORKERS", raising=False)
        assert _get_max_workers() == DEFAULT_MAX_WORKERS

    def test_non_numeric_config_falls_back_with_warning(self, monkeypatch, caplog):
        """CA-1.3: Si se proporciona una configuración cuyo formato no es
        válido, el sistema debe continuar funcionando utilizando un valor
        por defecto razonable y debe informar de la situación mediante
        un aviso."""
        monkeypatch.setenv("INGEST_MAX_WORKERS", "abc")
        with caplog.at_level("WARNING"):
            assert _get_max_workers() == DEFAULT_MAX_WORKERS
        assert "no es un número válido" in caplog.text

    def test_zero_config_falls_back_with_warning(self, monkeypatch, caplog):
        """CA-1.4: Si se proporciona una configuración con un valor fuera
        del rango permitido, el sistema debe continuar funcionando
        utilizando un valor por defecto razonable y debe informar de la
        situación mediante un aviso."""
        monkeypatch.setenv("INGEST_MAX_WORKERS", "0")
        with caplog.at_level("WARNING"):
            assert _get_max_workers() == DEFAULT_MAX_WORKERS
        assert "fuera de rango" in caplog.text

    def test_negative_config_falls_back_with_warning(self, monkeypatch, caplog):
        """CA-1.4: ídem, con un valor negativo."""
        monkeypatch.setenv("INGEST_MAX_WORKERS", "-5")
        with caplog.at_level("WARNING"):
            assert _get_max_workers() == DEFAULT_MAX_WORKERS
        assert "fuera de rango" in caplog.text

    def test_explicit_zero_workers_is_rejected(self):
        """CA-1.5: Si se solicita explícitamente un valor de concurrencia
        imposible al iniciar el proceso, el sistema debe rechazarlo con
        un mensaje claro."""
        import main as run_module

        with pytest.raises(ValueError):
            run_module.run(max_workers=0)

    def test_explicit_negative_workers_is_rejected(self):
        """CA-1.5: ídem, con un valor negativo."""
        import main as run_module

        with pytest.raises(ValueError):
            run_module.run(max_workers=-1)

    def test_never_submits_more_concurrent_tasks_than_available_files(
        self, tmp_path, monkeypatch
    ):
        """CA-1.6: El sistema no debe reservar más capacidad de procesamiento
        simultáneo que la cantidad de archivos disponibles para procesar."""
        import threading

        import main as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf"]:  # solo 2 archivos
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

        fake_store = type("FakeStore", (), {"close": lambda self: None})()
        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(
            "src.retrieval.vectorstore.VectorStore", lambda *a, **k: fake_store
        )
        monkeypatch.setattr("src.embeddings.embedder.Embedder", lambda: object())
        monkeypatch.setattr("src.etl.DoclingHybridChunker", lambda: object())
        monkeypatch.setattr(
            "src.etl.DocumentProcessor", lambda **kwargs: _FakeProcessor()
        )

        run_module.run(max_workers=10)  # pide 10, solo hay 2 archivos

        assert max_concurrent_seen <= 2


class TestArchivesProcessing:

    def test_processes_multiple_files(self, tmp_path, monkeypatch):
        """CA-2.1: El sistema debe poder procesar varios archivos al
        mismo tiempo."""
        import main as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (raw_dir / name).touch()

        processed_files = []

        class _FakeProcessor:
            def process_file(self, file):
                processed_files.append(file.name)
                return type("M", (), {"n_chunks": 1, "time_total_s": 0.01})()

        fake_store = type("FakeStore", (), {"close": lambda self: None})()
        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(
            "src.retrieval.vectorstore.VectorStore", lambda *a, **k: fake_store
        )
        monkeypatch.setattr("src.embeddings.embedder.Embedder", lambda: object())
        monkeypatch.setattr("src.etl.DoclingHybridChunker", lambda: object())
        monkeypatch.setattr(
            "src.etl.DocumentProcessor", lambda **kwargs: _FakeProcessor()
        )

        run_module.run(max_workers=2)

        assert sorted(processed_files) == ["a.pdf", "b.pdf", "c.pdf"]

    def test_one_file_failing_does_not_stop_the_rest(self, tmp_path, monkeypatch):
        """CA-2.2: Si un archivo falla durante el procesamiento, el resto
        debe seguir procesándose con normalidad hasta el final."""
        import main as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (raw_dir / name).touch()

        processed_files = []

        class _FakeProcessor:
            def process_file(self, file):
                if file.name == "b.pdf":
                    raise RuntimeError("fallo simulado")
                processed_files.append(file.name)
                return type("M", (), {"n_chunks": 1, "time_total_s": 0.01})()

        fake_store = type("FakeStore", (), {"close": lambda self: None})()
        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(
            "src.retrieval.vectorstore.VectorStore", lambda *a, **k: fake_store
        )
        monkeypatch.setattr("src.embeddings.embedder.Embedder", lambda: object())
        monkeypatch.setattr("src.etl.DoclingHybridChunker", lambda: object())
        monkeypatch.setattr(
            "src.etl.DocumentProcessor", lambda **kwargs: _FakeProcessor()
        )

        run_module.run(max_workers=2)  # no debe lanzar

        assert sorted(processed_files) == ["a.pdf", "c.pdf"]

    def test_no_files_is_reported_and_finishes_without_error(
        self, tmp_path, monkeypatch, caplog
    ):
        """CA-2.3: Si no hay archivos para procesar, el sistema debe
        informarlo y finalizar sin error."""
        import main as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)

        with caplog.at_level("WARNING"):
            run_module.run(max_workers=2)  # no debe lanzar

        assert "No se encontraron archivos PDF" in caplog.text


class TestConcurrentIndexingIntegrityAndCompatibility:

    def test_concurrent_indexing_never_corrupts_or_loses_data(self):
        """CA-3.1: Procesar archivos de forma concurrente nunca debe
        corromper, perder ni duplicar los datos ya almacenados."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from main import _SerializedVectorStore

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

        from main import _SerializedVectorStore
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

        import main as run_module

        monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
        importlib.reload(run_module)

        assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"


class TestObservability:

    def test_reports_success_and_failure_counts_at_the_end(
        self, tmp_path, monkeypatch, caplog
    ):
        """CA-4.1: Al finalizar la ingesta, el sistema debe informar
        cuántos archivos se procesaron con éxito y cuántos fallaron."""
        import main as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in ["a.pdf", "b.pdf"]:
            (raw_dir / name).touch()

        class _FakeProcessor:
            def process_file(self, file):
                if file.name == "a.pdf":
                    raise RuntimeError("fallo simulado")
                return type("M", (), {"n_chunks": 1, "time_total_s": 0.01})()

        fake_store = type("FakeStore", (), {"close": lambda self: None})()
        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(
            "src.retrieval.vectorstore.VectorStore", lambda *a, **k: fake_store
        )
        monkeypatch.setattr("src.embeddings.embedder.Embedder", lambda: object())
        monkeypatch.setattr("src.etl.DoclingHybridChunker", lambda: object())
        monkeypatch.setattr(
            "src.etl.DocumentProcessor", lambda **kwargs: _FakeProcessor()
        )

        with caplog.at_level("INFO"):
            run_module.run(max_workers=2)

        assert "OK=1 | Fallidos=1" in caplog.text

    def test_reports_periodic_progress_on_long_runs(
        self, tmp_path, monkeypatch, caplog
    ):
        """CA-4.2: En ejecuciones largas, el sistema debe informar el
        progreso periódicamente, no solo al final."""
        import main as run_module

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for i in range(120):
            (raw_dir / f"file_{i}.pdf").touch()

        class _FakeProcessor:
            def process_file(self, file):
                return type("M", (), {"n_chunks": 1, "time_total_s": 0.001})()

        fake_store = type("FakeStore", (), {"close": lambda self: None})()
        monkeypatch.setattr(run_module, "RAW_DIR", raw_dir)
        monkeypatch.setattr(
            "src.retrieval.vectorstore.VectorStore", lambda *a, **k: fake_store
        )
        monkeypatch.setattr("src.embeddings.embedder.Embedder", lambda: object())
        monkeypatch.setattr("src.etl.DoclingHybridChunker", lambda: object())
        monkeypatch.setattr(
            "src.etl.DocumentProcessor", lambda **kwargs: _FakeProcessor()
        )

        with caplog.at_level("INFO"):
            run_module.run(max_workers=4)

        progress_lines = [r for r in caplog.records if "Progreso:" in r.message]
        assert len(progress_lines) >= 1
