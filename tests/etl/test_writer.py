import json
from pathlib import Path


from src.etl.models import PageData, ProcessedDoc
from src.etl.writer import append_manifest_batch, save_doc

# ── Fixture base ──────────────────────────────────────────────────────────────


def _doc(doc_id: str = "ewrs/EWR_PM104_2012") -> ProcessedDoc:
    return ProcessedDoc(
        doc_id=doc_id,
        source_path=f"{doc_id}.pdf",
        doc_type="end_of_well_report",
        filename=doc_id.split("/")[-1],
        page_count=1,
        char_count=42,
        text="Pérdida en Quintuco a 2.450 metros.",
        pages=[PageData(page_num=1, text="Pérdida en Quintuco a 2.450 metros.")],
        extracted_at="2026-01-01T00:00:00+00:00",
        error=None,
    )


# ── Tests de save_doc ─────────────────────────────────────────────────────────


class TestSaveDoc:
    def test_creates_json_file(self, tmp_path):
        doc = _doc()
        out = save_doc(doc, tmp_path)
        assert out.exists()
        assert out.suffix == ".json"

    def test_json_is_valid(self, tmp_path):
        doc = _doc()
        out = save_doc(doc, tmp_path)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_json_contains_expected_fields(self, tmp_path):
        doc = _doc()
        out = save_doc(doc, tmp_path)
        data = json.loads(out.read_text(encoding="utf-8"))

        for field in (
            "doc_id",
            "source_path",
            "doc_type",
            "filename",
            "page_count",
            "char_count",
            "text",
            "pages",
            "extracted_at",
        ):
            assert field in data, f"campo faltante: {field}"

    def test_json_path_matches_doc_id(self, tmp_path):
        doc = _doc("partes_diarios/PD_PM104_20180514")
        out = save_doc(doc, tmp_path)
        assert out == tmp_path / "partes_diarios" / "PD_PM104_20180514.json"

    def test_creates_intermediate_dirs(self, tmp_path):
        doc = _doc("reservorio/sub/PVT_2024")
        out = save_doc(doc, tmp_path)
        assert out.parent.exists()

    def test_overwrites_existing_file(self, tmp_path):
        doc = _doc()
        save_doc(doc, tmp_path)
        doc2 = _doc()
        doc2.char_count = 999
        out = save_doc(doc2, tmp_path)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["char_count"] == 999

    def test_pages_are_serialized(self, tmp_path):
        doc = _doc()
        out = save_doc(doc, tmp_path)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data["pages"], list)
        assert data["pages"][0]["page_num"] == 1

    def test_returns_path_object(self, tmp_path):
        out = save_doc(_doc(), tmp_path)
        assert isinstance(out, Path)


# ── Tests de append_manifest_batch ──────────────────────────────────────────────────


class TestAppendManifest:
    def test_creates_manifest_file(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        append_manifest_batch(_doc(), manifest)
        assert manifest.exists()

    def test_each_line_is_valid_json(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        append_manifest_batch(_doc("ewrs/A"), manifest)
        append_manifest_batch(_doc("ewrs/B"), manifest)

        lines = manifest.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_manifest_excludes_text_and_pages(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        append_manifest_batch(_doc(), manifest)

        data = json.loads(manifest.read_text(encoding="utf-8").strip())
        assert "text" not in data
        assert "pages" not in data

    def test_manifest_includes_metadata_fields(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        append_manifest_batch(_doc(), manifest)

        data = json.loads(manifest.read_text(encoding="utf-8").strip())
        for field in (
            "doc_id",
            "doc_type",
            "filename",
            "page_count",
            "char_count",
            "extracted_at",
        ):
            assert field in data

    def test_appends_multiple_docs(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        for i in range(5):
            append_manifest_batch(_doc(f"ewrs/DOC_{i}"), manifest)

        lines = manifest.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_creates_parent_dirs(self, tmp_path):
        manifest = tmp_path / "subdir" / "manifest.jsonl"
        append_manifest_batch(_doc(), manifest)
        assert manifest.exists()
