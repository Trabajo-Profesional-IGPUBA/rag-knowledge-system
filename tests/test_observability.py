import json
import logging
from pathlib import Path
from src.observability import PipelineMetrics, Timer, setup_logging


class TestPipelineMetrics:
    def test_default_values(self):
        m = PipelineMetrics()
        assert m.docs_ok == 0
        assert m.docs_error == 0
        assert m.chunks_generated == 0
        assert m.vectors_indexed == 0

    def test_finish_sets_finished_at(self):
        m = PipelineMetrics()
        assert m.finished_at == ""
        m.finish()
        assert m.finished_at != ""
        assert "T" in m.finished_at

    def test_to_dict_returns_dict(self):
        m = PipelineMetrics()
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "docs_ok" in d
        assert "chunks_generated" in d

    def test_save_creates_json(self, tmp_path):
        m = PipelineMetrics()
        m.docs_ok = 5
        m.chunks_generated = 42
        path = tmp_path / "metrics.json"
        m.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["docs_ok"] == 5
        assert data["chunks_generated"] == 42

    def test_save_creates_parent_dirs(self, tmp_path):
        m = PipelineMetrics()
        path = tmp_path / "sub" / "dir" / "metrics.json"
        m.save(path)
        assert path.exists()


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
