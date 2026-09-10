import pytest

from main import DEFAULT_MAX_WORKERS, _get_max_workers

"""
Cubren "Paralelizar la ingesta de documentos (ETL)":
  - Configuración de la concurrencia-> CA-1.1 a CA-1.6
"""


class TestConfigurableConcurrency:

    def test_allows_configuring_number_of_concurrent_files(self, monkeypatch):
        """CA-1.1: El sistema debe permitir configurar cuántos archivos se
        procesan al mismo tiempo."""
        monkeypatch.setenv("INGEST_MAX_WORKERS", "3")
        assert _get_max_workers() == 3


class TestDefaultConcurrencyFallback:

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
