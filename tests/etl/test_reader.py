from pathlib import Path

from fpdf import FPDF

from src.etl.reader import extract, _clean

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _write_pdf(tmp_path: Path, subfolder: str, filename: str, text: str) -> Path:
    folder = tmp_path / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / filename

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, text)
    pdf.output(str(out))
    return out


# ── Tests de _clean ───────────────────────────────────────────────────────────


class TestClean:
    def test_collapses_multiple_newlines(self):
        result = _clean("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strips_edges(self):
        assert _clean("  hola  \n") == "hola"

    def test_empty_string(self):
        assert _clean("") == ""


# ── Tests de extract ──────────────────────────────────────────────────────────


class TestExtract:
    def test_extracts_text_from_valid_pdf(self, tmp_path):
        pdf_path = _write_pdf(
            tmp_path,
            "ewrs",
            "EWR_PM104_2012.pdf",
            "Pérdida de circulación severa en Quintuco a 2.450 metros.",
        )
        doc = extract(pdf_path, tmp_path)

        assert doc.ok()
        assert "Quintuco" in doc.text
        assert doc.page_count == 1
        assert doc.char_count > 0
        assert doc.error is None

    def test_doc_id_is_relative_path_without_extension(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, "ewrs", "EWR_PM104_2012.pdf", "contenido")
        doc = extract(pdf_path, tmp_path)
        assert doc.doc_id == "ewrs/EWR_PM104_2012"

    def test_doc_type_resolved_from_folder(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, "partes_diarios", "PD_X.pdf", "parte")
        doc = extract(pdf_path, tmp_path)
        assert doc.doc_type == "parte_diario"

    def test_unknown_folder_gets_desconocido_type(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, "misc", "archivo.pdf", "texto")
        doc = extract(pdf_path, tmp_path)
        assert doc.doc_type == "desconocido"

    def test_pages_list_matches_page_count(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, "ewrs", "multi.pdf", "texto de prueba")
        doc = extract(pdf_path, tmp_path)
        assert len(doc.pages) == doc.page_count

    def test_pages_have_correct_page_num(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, "ewrs", "paged.pdf", "página uno")
        doc = extract(pdf_path, tmp_path)
        assert doc.pages[0].page_num == 1

    def test_nonexistent_file_returns_error(self, tmp_path):
        fake = tmp_path / "ewrs" / "noexiste.pdf"
        (tmp_path / "ewrs").mkdir()
        doc = extract(fake, tmp_path)
        assert not doc.ok()
        assert doc.error is not None
        assert doc.char_count == 0
        assert doc.text == ""

    def test_corrupted_file_returns_error(self, tmp_path):
        folder = tmp_path / "ewrs"
        folder.mkdir()
        bad = folder / "corrupt.pdf"
        bad.write_bytes(b"esto no es un pdf valido %%%")
        doc = extract(bad, tmp_path)
        assert not doc.ok()

    def test_extracted_at_is_set(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, "ewrs", "ts.pdf", "texto")
        doc = extract(pdf_path, tmp_path)
        assert doc.extracted_at
        assert "T" in doc.extracted_at

    def test_filename_without_extension(self, tmp_path):
        pdf_path = _write_pdf(tmp_path, "fallas_bes", "DIFA_PM104_2021.pdf", "falla")
        doc = extract(pdf_path, tmp_path)
        assert doc.filename == "DIFA_PM104_2021"
