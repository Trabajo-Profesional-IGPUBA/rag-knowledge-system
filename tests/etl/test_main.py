from main import DEFAULT_MAX_WORKERS, _get_max_workers


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
