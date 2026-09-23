from unittest.mock import MagicMock

import pytest

from src.etl.chunker import _PATTERNS, Chunk, DoclingHybridChunker


def make_chunk(**kwargs) -> Chunk:
    """Builds a Chunk with sensible defaults for unit tests."""
    defaults = {
        "text": "Loss of circulation detected in Quintuco formation.",
        "contextualized_text": "Section: Operational Incidents\nLoss of circulation detected in Quintuco.",
        "source_file": "data/ewr/EWR_PM104_2012.pdf",
        "source_hash": "abc123",
        "page": 1,
        "section": "Operational Incidents",
        "chunk_index": 0,
        "well": "PM-104",
    }
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

    def test_chunks_never_exceed_max_tokens(self, tmp_path):
        """CA-1.2: El tamaño máximo de texto que puede procesar el modelo debe
        coincidir con el tamaño que se usa para dividir los documentos en
        fragmentos, de forma que ningún fragmento se corte sin que el sistema
        lo sepa."""
        MAX_TOKENS = 500

        # Texto largo a propósito, sin cortes naturales de párrafo antes de los
        # 500 tokens, para forzar al chunker a ejercitar el límite real.
        long_text = "Pérdida de circulación en formación Quintuco. " * 200
        sample_file = tmp_path / "sample.md"
        sample_file.write_text(f"# Sección: Incidentes Operativos\n\n{long_text}")

        chunker = DoclingHybridChunker()
        for chunk in chunker.chunk(sample_file):
            token_count = len(chunker._chunker.tokenizer.tokenizer.encode(chunk.text))
            assert token_count <= MAX_TOKENS


class TestPatterns:
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


class TestWellDetection:
    """CA-1.1: Dado un documento que contiene un identificador de pozo en alguno
    de los formatos reconocidos, cuando se procesa el texto, entonces el sistema
    debe detectar y extraer ese identificador."""

    @pytest.mark.parametrize(
        "text",
        [
            "Pozo: CH-88",
            "pozo CH-88",
            "POZO: YPF-001",
            "pozo: PI-05",
            "Pozo: LL 112",
            "pozo LL 112",
            "Pozo: ch88",
            "pozo CH88",
            "El pozo CH-88 presentó pérdida de circulación.",
            "Intervención en pozo LL 112 durante fase intermedia.",
        ],
    )
    def test_detects_well_identifier(self, text):
        """CA-1.1: El sistema detecta el identificador en el formato dado."""
        meta = DoclingHybridChunker.extract_chunk_metadata(text)
        assert (
            meta.well is not None
        ), f"No se detectó ningún identificador de pozo en: {text!r}"

    CANONICAL_CASES = [
        ("Pozo: CH-88", "CH-88"),
        ("pozo ch-88", "CH-88"),
        ("pozo CH88", "CH-88"),
        ("Pozo: ch 88", "CH-88"),
        ("POZO: Ch-88", "CH-88"),
        ("Pozo: LL-112", "LL-112"),
        ("pozo ll 112", "LL-112"),
        ("Pozo: ll112", "LL-112"),
        ("POZO: LL112", "LL-112"),
    ]

    """CA-1.2: Dado un identificador de pozo extraído en cualquier variante de
    formato, cuando se normaliza, entonces debe convertirse a un formato
    canónico único."""

    @pytest.mark.parametrize("text, canonical", CANONICAL_CASES)
    def test_normalizes_to_canonical_form(self, text, canonical):
        """CA-1.2: Variantes del mismo identificador producen siempre el mismo valor canónico."""
        meta = DoclingHybridChunker.extract_chunk_metadata(text)
        assert meta.well == canonical, (
            f"Se esperaba '{canonical}' pero se obtuvo '{meta.well}' "
            f"para el texto: {text!r}"
        )

    def test_same_well_different_formats_produce_identical_output(self):
        """CA-1.2: Todas las variantes del mismo pozo producen exactamente el mismo string."""
        variants = [
            text for text, canonical in self.CANONICAL_CASES if canonical == "CH-88"
        ]
        results = {
            DoclingHybridChunker.extract_chunk_metadata(text).well for text in variants
        }
        assert len(results) == 1, (
            f"Se esperaba un único valor canónico para CH-88, "
            f"pero se obtuvieron {len(results)} variantes: {results}"
        )

    """CA-1.3: Dado un documento que no contiene ningún identificador de pozo
    reconocible, cuando se procesa el texto, entonces el sistema debe manejar
    el caso sin fallar, retornando None."""

    @pytest.mark.parametrize(
        "text",
        [
            "No well reference here.",
            "The field has good production.",
            "Temperature: 120°C",
            "Cementación ejecutada con retorno total a superficie.",
            "",
            "   ",
        ],
    )
    def test_returns_none_when_no_well_present(self, text):
        """CA-1.3: El sistema retorna None sin fallar cuando no hay identificador."""
        meta = DoclingHybridChunker.extract_chunk_metadata(text)
        assert meta.well is None, (
            f"Se esperaba None pero se obtuvo '{meta.well}' " f"para el texto: {text!r}"
        )
