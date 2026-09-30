import pytest
from src.etl.chunk_meta import extract_wells


class TestChunkMetaFromText:
    """
    Tests de ChunkMeta.from_text — lógica de extracción de metadatos de dominio.
    Estos tests son independientes de Docling y del chunker.
    """

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
        wells = extract_wells(text)
        assert (
            len(wells) > 0
        ), f"No se detectó ningún identificador de pozo en: {text!r}"

    CANONICAL_CASES = [
        ("Pozo: CH-88", "CH-88"),
        ("pozo ch-88", "CH-88"),
        ("pozo CH88", "CH-88"),
        ("Pozo: ch 88", "CH-88"),
        ("POZO: Ch-88", "CH-88"),
    ]

    @pytest.mark.parametrize("text, canonical", CANONICAL_CASES)
    def test_normalizes_to_canonical_form(self, text, canonical):
        """CA-1.2: Variantes del mismo identificador producen siempre el mismo valor canónico."""
        wells = extract_wells(text)
        assert wells[0] == canonical, (
            f"Se esperaba '{canonical}' pero se obtuvo '{wells}' "
            f"para el texto: {text!r}"
        )

    def test_same_well_different_formats_produce_identical_output(self):
        """CA-1.2: Todas las variantes del mismo pozo producen exactamente el mismo string."""
        results = {extract_wells(text)[0] for (text, _) in self.CANONICAL_CASES}
        assert len(results) == 1, (
            f"Se esperaba un único valor canónico para CH-88, "
            f"pero se obtuvieron {len(results)} variantes: {results}"
        )

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
        wells = extract_wells(text)
        assert len(wells) == 0, (
            f"Se esperaba None pero se obtuvo '{wells}' " f"para el texto: {text!r}"
        )
