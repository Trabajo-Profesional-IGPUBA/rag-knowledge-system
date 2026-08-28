from src.etl.cleaner import normalize, extract_metadata_hints


class TestNormalize:
    def test_empty_string(self):
        assert normalize("") == ""

    def test_collapses_multiple_newlines(self):
        result = normalize("párrafo 1\n\n\n\npárrafo 2")
        assert result == "párrafo 1\n\npárrafo 2"

    def test_strips_whitespace(self):
        assert normalize("  hola  \n") == "hola"

    def test_removes_control_chars(self):
        result = normalize("texto\x00con\x01control")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_normalizes_dashes(self):
        result = normalize("temperatura \u2013 presión \u2014 caudal")
        assert "\u2013" not in result
        assert "\u2014" not in result
        assert "-" in result

    def test_normalizes_quotes(self):
        # Comillas tipográficas unicode → comillas ASCII
        tipograficas = "\u201ccomillas tipográficas\u201d"
        result = normalize(tipograficas)
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"' in result

    def test_removes_ocr_artifacts(self):
        result = normalize("texto|||||más texto")
        assert "||||" not in result

    def test_preserves_technical_units(self):
        text = "presión de 1.200 psi y caudal de 50 m³/día"
        result = normalize(text)
        assert "psi" in result
        assert "m³/día" in result

    def test_unicode_nfc(self):
        import unicodedata

        composed = unicodedata.normalize("NFC", "é")
        decomposed = unicodedata.normalize("NFD", "é")
        assert normalize(decomposed) == composed

    def test_multiline_preserved(self):
        result = normalize("línea 1\nlínea 2\n\npárrafo 2")
        assert "línea 1" in result
        assert "línea 2" in result
        assert "párrafo 2" in result


class TestExtractMetadataHints:
    def test_extracts_pozo(self):
        hints = extract_metadata_hints("POZO: PM-104\ncontenido")
        assert hints.get("pozo") == "PM-104"

    def test_extracts_año(self):
        hints = extract_metadata_hints("AÑO: 2021\ncontenido")
        assert hints.get("año") == "2021"

    def test_returns_empty_dict_when_no_hints(self):
        hints = extract_metadata_hints("texto sin metadata estructurada")
        assert hints == {}

    def test_case_insensitive(self):
        hints = extract_metadata_hints("pozo: LL-205")
        assert "pozo" in hints
