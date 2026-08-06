from unittest.mock import MagicMock, patch
from src.etl.ocr import extract_text_from_page, needs_ocr, MIN_CHARS


class TestExtractTextFromPage:
    def _mock_page(self, text: str):
        page = MagicMock()
        page.extract_text.return_value = text
        return page

    def test_digital_page_returns_text_without_ocr(self):
        page = self._mock_page("A" * (MIN_CHARS + 10))
        text, used_ocr = extract_text_from_page(page)
        assert text == "A" * (MIN_CHARS + 10)
        assert used_ocr is False

    def test_empty_page_triggers_ocr_attempt(self):
        page = self._mock_page("")
        with patch("src.etl.ocr._ocr_page", return_value="texto ocr") as mock_ocr:
            text, used_ocr = extract_text_from_page(page)
            assert used_ocr is True
            mock_ocr.assert_called_once_with(page)

    def test_short_text_triggers_ocr(self):
        page = self._mock_page("abc")
        with patch("src.etl.ocr._ocr_page", return_value="texto largo via ocr"):
            text, used_ocr = extract_text_from_page(page)
            assert used_ocr is True

    def test_exact_threshold_no_ocr(self):
        page = self._mock_page("x" * MIN_CHARS)
        text, used_ocr = extract_text_from_page(page)
        assert used_ocr is False

    def test_returns_ocr_text_when_digital_empty(self):
        page = self._mock_page("")
        with patch("src.etl.ocr._ocr_page", return_value="resultado ocr"):
            text, used_ocr = extract_text_from_page(page)
            assert text == "resultado ocr"


class TestNeedsOcr:
    def test_returns_bool(self, tmp_path):
        from fpdf import FPDF
        pdf_path = tmp_path / "test.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, "Texto largo digital " * 20)
        pdf.output(str(pdf_path))

        result = needs_ocr(pdf_path)
        assert isinstance(result, bool)

    def test_digital_pdf_does_not_need_ocr(self, tmp_path):
        from fpdf import FPDF
        pdf_path = tmp_path / "digital.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, "Informe técnico completo con mucho texto de contenido " * 10)
        pdf.output(str(pdf_path))

        assert needs_ocr(pdf_path) is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert needs_ocr(tmp_path / "noexiste.pdf") is False
