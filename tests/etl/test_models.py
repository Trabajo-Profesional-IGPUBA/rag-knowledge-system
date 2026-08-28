from src.etl.models import PageData, ProcessedDoc


def _make_doc(**overrides) -> ProcessedDoc:
    defaults = {
        "doc_id": "ewrs/EWR_PM104_2012",
        "source_path": "ewrs/EWR_PM104_2012.pdf",
        "doc_type": "end_of_well_report",
        "filename": "EWR_PM104_2012",
        "page_count": 2,
        "char_count": 500,
        "text": "Texto de ejemplo.",
        "pages": [PageData(page_num=1, text="Página 1")],
        "extracted_at": "2026-01-01T00:00:00+00:00",
        "error": None,
    }
    defaults.update(overrides)
    return ProcessedDoc(**defaults)


class TestPageData:
    def test_fields(self):
        p = PageData(page_num=3, text="Hola mundo")
        assert p.page_num == 3
        assert p.text == "Hola mundo"


class TestProcessedDoc:
    def test_ok_when_no_error(self):
        doc = _make_doc(error=None)
        assert doc.ok() is True

    def test_not_ok_when_error(self):
        doc = _make_doc(error="PdfReadError: EOF marker not found")
        assert doc.ok() is False

    def test_fields_stored_correctly(self):
        pages = [PageData(1, "p1"), PageData(2, "p2")]
        doc = _make_doc(page_count=2, pages=pages, char_count=4)
        assert doc.page_count == 2
        assert len(doc.pages) == 2
        assert doc.pages[0].text == "p1"

    def test_error_is_none_by_default(self):
        doc = _make_doc()
        assert doc.error is None
