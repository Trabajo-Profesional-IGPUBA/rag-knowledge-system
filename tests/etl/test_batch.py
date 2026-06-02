from pathlib import Path

from fpdf import FPDF

from src.etl.batch import discover, run

# ── Helper para crear PDFs de prueba ─────────────────────────────────────────


def _make_pdf(path: Path, text: str = "Texto de prueba."):
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, text)
    pdf.output(str(path))


def _setup_raw(tmp_path: Path, filenames: list[str]) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    mani = tmp_path / "manifest.jsonl"
    for name in filenames:
        _make_pdf(raw / name)
    return raw, proc, mani


# ── Tests de discover ─────────────────────────────────────────────────────────


class TestDiscover:
    def test_returns_all_pdfs_when_none_processed(self, tmp_path):
        raw, proc, _ = _setup_raw(
            tmp_path, ["ewrs/A.pdf", "ewrs/B.pdf", "partes_diarios/C.pdf"]
        )
        pending, skipped = discover(raw, proc, incremental=True)
        assert len(pending) == 3
        assert skipped == 0

    def test_skips_already_processed(self, tmp_path):
        raw, proc, _ = _setup_raw(tmp_path, ["ewrs/A.pdf", "ewrs/B.pdf"])
        already = proc / "ewrs" / "A.json"
        already.parent.mkdir(parents=True, exist_ok=True)
        already.write_text("{}")

        pending, skipped = discover(raw, proc, incremental=True)
        assert len(pending) == 1
        assert skipped == 1
        assert pending[0].name == "B.pdf"

    def test_no_incremental_returns_all(self, tmp_path):
        raw, proc, _ = _setup_raw(tmp_path, ["ewrs/A.pdf", "ewrs/B.pdf"])
        already = proc / "ewrs" / "A.json"
        already.parent.mkdir(parents=True, exist_ok=True)
        already.write_text("{}")

        pending, skipped = discover(raw, proc, incremental=False)
        assert len(pending) == 2
        assert skipped == 0

    def test_empty_raw_dir(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        proc = tmp_path / "processed"
        pending, skipped = discover(raw, proc)
        assert pending == []
        assert skipped == 0


# ── Tests de run ──────────────────────────────────────────────────────────────


class TestRun:
    def test_processes_all_pdfs(self, tmp_path):
        raw, proc, mani = _setup_raw(
            tmp_path,
            ["ewrs/EWR_A.pdf", "partes_diarios/PD_B.pdf", "fallas_bes/DIFA_C.pdf"],
        )
        result = run(raw, proc, mani, batch_size=2)

        assert result.total_ok == 3
        assert result.total_errors == 0

    def test_creates_json_for_each_pdf(self, tmp_path):
        raw, proc, mani = _setup_raw(tmp_path, ["ewrs/EWR_A.pdf", "ewrs/EWR_B.pdf"])
        run(raw, proc, mani)

        assert (proc / "ewrs" / "EWR_A.json").exists()
        assert (proc / "ewrs" / "EWR_B.json").exists()

    def test_creates_manifest(self, tmp_path):
        raw, proc, mani = _setup_raw(tmp_path, ["ewrs/X.pdf"])
        run(raw, proc, mani)
        assert mani.exists()
        assert mani.stat().st_size > 0

    def test_manifest_has_one_line_per_doc(self, tmp_path):
        raw, proc, mani = _setup_raw(
            tmp_path, ["ewrs/A.pdf", "ewrs/B.pdf", "ewrs/C.pdf"]
        )
        run(raw, proc, mani)
        lines = [line for line in mani.read_text().splitlines() if line.strip()]
        assert len(lines) == 3

    def test_incremental_skips_existing(self, tmp_path):
        raw, proc, mani = _setup_raw(tmp_path, ["ewrs/A.pdf", "ewrs/B.pdf"])
        # Pre-procesar A
        already = proc / "ewrs" / "A.json"
        already.parent.mkdir(parents=True, exist_ok=True)
        already.write_text('{"doc_id": "ewrs/A"}')

        result = run(raw, proc, mani, incremental=True)
        assert result.total_ok == 1  # solo B
        assert result.total_skipped == 1  # A ya existía

    def test_dry_run_does_not_write_files(self, tmp_path):
        raw, proc, mani = _setup_raw(tmp_path, ["ewrs/A.pdf"])
        run(raw, proc, mani, dry_run=True)
        assert not (proc / "ewrs" / "A.json").exists()
        assert not mani.exists()

    def test_corrupt_pdf_counted_as_error(self, tmp_path):
        raw = tmp_path / "raw" / "ewrs"
        raw.mkdir(parents=True)
        bad = raw / "corrupt.pdf"
        bad.write_bytes(b"no soy un pdf valido %%%")

        proc = tmp_path / "processed"
        mani = tmp_path / "manifest.jsonl"
        result = run(raw.parent, proc, mani)

        assert result.total_errors == 1
        assert result.total_ok == 0

    def test_result_elapsed_is_positive(self, tmp_path):
        raw, proc, mani = _setup_raw(tmp_path, ["ewrs/A.pdf"])
        result = run(raw, proc, mani)
        assert result.elapsed_sec > 0

    def test_batch_size_respected(self, tmp_path):
        """Con batch_size=1 debe procesar de a un PDF y terminar igual."""
        raw, proc, mani = _setup_raw(
            tmp_path, ["ewrs/A.pdf", "ewrs/B.pdf", "ewrs/C.pdf"]
        )
        result = run(raw, proc, mani, batch_size=1)
        assert result.total_ok == 3

    def test_empty_raw_returns_zero(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        proc = tmp_path / "processed"
        mani = tmp_path / "manifest.jsonl"
        result = run(raw, proc, mani)
        assert result.total_ok == 0
        assert result.total_errors == 0
