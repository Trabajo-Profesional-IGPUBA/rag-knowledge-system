from pathlib import Path

from fpdf import FPDF

from src.etl.batch import run


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


class TestRun:

    def test_processes_all_pdfs(self, tmp_path):
        raw, proc, mani = _setup_raw(
            tmp_path,
            [
                "ewrs/EWR_A.pdf",
                "partes_diarios/PD_B.pdf",
                "fallas_bes/DIFA_C.pdf",
            ],
        )

        result = run(raw, proc, mani)

        assert result.total_ok == 3
        assert result.total_errors == 0

    def test_creates_json_for_each_pdf(self, tmp_path):
        raw, proc, mani = _setup_raw(
            tmp_path,
            ["ewrs/EWR_A.pdf", "ewrs/EWR_B.pdf"],
        )

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
            tmp_path,
            ["ewrs/A.pdf", "ewrs/B.pdf", "ewrs/C.pdf"],
        )

        run(raw, proc, mani)

        lines = [line for line in mani.read_text().splitlines() if line.strip()]
        assert len(lines) == 3

    def test_incremental_skips_existing(self, tmp_path):
        raw, proc, mani = _setup_raw(tmp_path, ["ewrs/A.pdf", "ewrs/B.pdf"])

        already = proc / "ewrs" / "A.json"
        already.parent.mkdir(parents=True, exist_ok=True)
        already.write_text("{}")

        result = run(raw, proc, mani, incremental=True)

        assert result.total_ok == 1
        assert result.total_skipped == 1

    def test_corrupt_pdf_counted_as_error(self, tmp_path):
        raw = tmp_path / "raw" / "ewrs"
        raw.mkdir(parents=True)

        bad = raw / "corrupt.pdf"
        bad.write_bytes(b"no soy un pdf valido %%%")

        proc = tmp_path / "processed"
        mani = tmp_path / "manifest.jsonl"

        result = run(raw.parent, proc, mani)

        assert result.total_errors >= 1

    def test_result_elapsed_is_positive(self, tmp_path):
        raw, proc, mani = _setup_raw(tmp_path, ["ewrs/A.pdf"])

        result = run(raw, proc, mani)

        assert result.elapsed_sec > 0

    def test_empty_raw_returns_zero(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()

        proc = tmp_path / "processed"
        mani = tmp_path / "manifest.jsonl"

        result = run(raw, proc, mani)

        assert result.total_ok == 0
        assert result.total_errors == 0
