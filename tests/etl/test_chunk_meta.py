from src.etl.chunk_meta import ChunkMeta, _PATTERNS

import pytest


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


class TestChunkMetaFromText:
    """
    Tests de ChunkMeta.from_text — lógica de extracción de metadatos de dominio.
    Estos tests son independientes de Docling y del chunker.
    """

    # ── Sección ───────────────────────────────────────────────────────────

    def test_extracts_section(self):
        text = "Sección: Incidentes Operativos\nLoss of circulation."
        meta = ChunkMeta.from_text(text)
        assert meta.section == "Incidentes Operativos"

    def test_returns_none_when_no_section(self):
        text = "Pozo: PM-104\nLoss of circulation detected."
        assert ChunkMeta.from_text(text).section is None

    # ── Well — CA-1.1: detección ──────────────────────────────────────────

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
        meta = ChunkMeta.from_text(text)
        assert (
            meta.well is not None
        ), f"No se detectó ningún identificador de pozo en: {text!r}"

    # ── Well — CA-1.2: normalización ─────────────────────────────────────

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

    @pytest.mark.parametrize("text, canonical", CANONICAL_CASES)
    def test_normalizes_to_canonical_form(self, text, canonical):
        """CA-1.2: Variantes del mismo identificador producen siempre el mismo valor canónico."""
        meta = ChunkMeta.from_text(text)
        assert meta.well == canonical, (
            f"Se esperaba '{canonical}' pero se obtuvo '{meta.well}' "
            f"para el texto: {text!r}"
        )

    def test_same_well_different_formats_produce_identical_output(self):
        """CA-1.2: Todas las variantes del mismo pozo producen exactamente el mismo string."""
        variants = [
            text for text, canonical in self.CANONICAL_CASES if canonical == "CH-88"
        ]
        results = {ChunkMeta.from_text(text).well for text in variants}
        assert len(results) == 1, (
            f"Se esperaba un único valor canónico para CH-88, "
            f"pero se obtuvieron {len(results)} variantes: {results}"
        )

    # ── Well — CA-1.3: ausencia ───────────────────────────────────────────

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
        meta = ChunkMeta.from_text(text)
        assert meta.well is None, (
            f"Se esperaba None pero se obtuvo '{meta.well}' " f"para el texto: {text!r}"
        )

    def test_does_not_raise_on_empty_string(self):
        """CA-1.3: El sistema no lanza excepción con texto vacío."""
        meta = ChunkMeta.from_text("")
        assert meta.well is None
