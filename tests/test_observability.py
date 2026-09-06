from src.observability import Timer, setup_logging


class TestTimer:
    def test_measures_elapsed(self):
        import time

        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_elapsed_is_float(self):
        with Timer() as t:
            pass
        assert isinstance(t.elapsed, float)


class TestSetupLogging:
    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "logs"
        setup_logging(log_dir)
        assert log_dir.exists()

    def test_log_file_created(self, tmp_path):
        log_dir = tmp_path / "logs"
        setup_logging(log_dir)
        assert (log_dir / "rag_pipeline.log").exists()
