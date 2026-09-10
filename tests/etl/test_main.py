from main import _get_max_workers


class TestConfigurableConcurrency:

    def test_allows_configuring_number_of_concurrent_files(self, monkeypatch):
        """CA-1.1: El sistema debe permitir configurar cuántos archivos se
        procesan al mismo tiempo."""
        monkeypatch.setenv("INGEST_MAX_WORKERS", "3")
        assert _get_max_workers() == 3
