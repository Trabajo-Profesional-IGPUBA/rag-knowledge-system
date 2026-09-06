from unittest.mock import MagicMock

import pytest

from src.etl.chunker import Chunk, DoclingHybridChunker, _PATTERNS


def make_chunk(**kwargs) -> Chunk:
    """Builds a Chunk with sensible defaults for unit tests."""
    defaults = dict(
        text="Loss of circulation detected in Quintuco formation.",
        contextualized_text="Section: Operational Incidents\nLoss of circulation detected in Quintuco.",
        source_file="data/ewr/EWR_PM104_2012.pdf",
        source_hash="abc123",
        page=1,
        section="Operational Incidents",
        chunk_index=0,
        well="PM-104",
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


class TestChunk:

    def test_chunk_id_format(self):
        chunk = make_chunk(source_hash="abc123", chunk_index=5)
        assert chunk.chunk_id == "abc123::5"

    def test_chunk_id_unique_per_index(self):
        c1 = make_chunk(source_hash="abc123", chunk_index=0)
        c2 = make_chunk(source_hash="abc123", chunk_index=1)
        assert c1.chunk_id != c2.chunk_id

    def test_char_count(self):
        chunk = make_chunk(text="hello world")
        assert chunk.char_count == 11

    def test_meta_contains_required_fields(self):
        chunk = make_chunk()
        for field in ["well", "section", "source_file", "source_hash", "page"]:
            assert field in chunk.meta, f"Required field missing from meta: {field}"

    def test_meta_values_correct(self):
        chunk = make_chunk(well="PM-104", section="Incidents", page=47)
        assert chunk.meta["well"] == "PM-104"
        assert chunk.meta["section"] == "Incidents"
        assert chunk.meta["page"] == 47

    def test_meta_accepts_none(self):
        chunk = make_chunk(well=None, section=None, page=None)
        assert chunk.meta["well"] is None
        assert chunk.meta["section"] is None
        assert chunk.meta["page"] is None


class TestPatterns:

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Pozo: PM-104", "PM-104"),
            ("pozo PM-104", "PM-104"),
            ("POZO: YPF-001", "YPF-001"),
            ("Pozo: 6506/3-1", "6506/3-1"),
            ("pozo: PI-05", "PI-05"),
        ],
    )
    def test_well_matches_valid_formats(self, text, expected):
        match = _PATTERNS["well"].search(text)
        assert match is not None, f"Did not match: {text!r}"
        assert match.group(1).strip() == expected

    @pytest.mark.parametrize(
        "text",
        [
            "No well reference here.",
            "The field has good production.",
            "Temperature: 120°C",
        ],
    )
    def test_well_no_false_positive(self, text):
        assert _PATTERNS["well"].search(text) is None

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Sección: Incidentes Operativos", "Incidentes Operativos"),
            ("sección: Cementación", "Cementación"),
            ("SECCION: Datos Generales del Pozo", "Datos Generales del Pozo"),
        ],
    )
    def test_section_matches_valid_formats(self, text, expected):
        match = _PATTERNS["section"].search(text)
        assert match is not None, f"Did not match: {text!r}"
        assert match.group(1).strip() == expected

    def test_section_no_false_positive(self):
        text = "The well has good production in the upper interval."
        assert _PATTERNS["section"].search(text) is None


class TestExtractChunkMetadata:

    def test_extracts_well_and_section(self):
        text = "Sección: Incidentes Operativos\nPozo: PM-104\nLoss of circulation."
        meta = DoclingHybridChunker.extract_chunk_metadata(text)
        assert meta.well == "PM-104"
        assert meta.section == "Incidentes Operativos"

    def test_returns_none_when_no_well(self):
        text = "Sección: Cementación\nCement plug was set."
        meta = DoclingHybridChunker.extract_chunk_metadata(text)
        assert meta.well is None

    def test_returns_none_when_no_section(self):
        text = "Pozo: PM-104\nLoss of circulation detected."
        meta = DoclingHybridChunker.extract_chunk_metadata(text)
        assert meta.section is None

    def test_returns_none_on_empty_text(self):
        meta = DoclingHybridChunker.extract_chunk_metadata("")
        assert meta.well is None
        assert meta.section is None

    def test_extracts_from_contextualized_text(self):
        """
        The chunker extracts metadata from contextualized_text, not plain text.
        Verifies that a well name in the heading path is correctly detected.
        """
        text = "1.2 Well Data\nPozo: 6506/3-1\nTotal depth: 3200 mdf."
        meta = DoclingHybridChunker.extract_chunk_metadata(text)
        assert meta.well == "6506/3-1"


class TestExtractPageNo:

    def _make_raw_chunk(self, page_no: int | None) -> MagicMock:
        """Builds a Docling chunk mock with the provenance structure."""
        chunk = MagicMock()
        if page_no is not None:
            prov = MagicMock()
            prov.page_no = page_no
            doc_item = MagicMock()
            doc_item.prov = [prov]
            chunk.meta.doc_items = [doc_item]
        else:
            chunk.meta.doc_items = []
        return chunk

    def test_extracts_page_number(self):
        raw = self._make_raw_chunk(page_no=47)
        assert DoclingHybridChunker.extract_page_no(raw) == 47

    def test_returns_none_without_provenance(self):
        raw = self._make_raw_chunk(page_no=None)
        assert DoclingHybridChunker.extract_page_no(raw) is None

    def test_returns_none_with_empty_doc_items(self):
        raw = MagicMock()
        raw.meta.doc_items = []
        assert DoclingHybridChunker.extract_page_no(raw) is None
