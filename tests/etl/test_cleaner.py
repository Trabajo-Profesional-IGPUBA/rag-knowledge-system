from src.etl.cleaner import extract_metadata_hints, normalize


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

    def test_ca11_removes_null_bytes(self):
        """CA-1.1: El sistema debe eliminar caracteres de control del texto extraído."""
        result = normalize("texto\x00con\x07bytes\x1fnulos")
        assert "\x00" not in result
        assert "\x07" not in result
        assert "\x1f" not in result

    def test_ca11_removes_pipe_ocr_artifacts(self):
        """CA-1.1: El sistema debe eliminar artefactos comunes de OCR como secuencias de pipes."""
        result = normalize("dato|||||||valor")
        assert "|||" not in result

    def test_ca11_removes_underscore_artifacts(self):
        """CA-1.1: El sistema debe eliminar secuencias largas de guiones bajos."""
        result = normalize("título____________________valor")
        assert "____" not in result

    def test_ca12_normalizes_em_dash(self):
        """CA-1.2: El sistema debe reemplazar guiones largos por guiones estándar."""
        result = normalize("presión\u2014caudal")
        assert "\u2014" not in result
        assert "-" in result

    def test_ca12_normalizes_en_dash(self):
        """CA-1.2: El sistema debe reemplazar guiones medios por guiones estándar."""
        result = normalize("valor\u2013otro")
        assert "\u2013" not in result
        assert "-" in result

    def test_ca12_normalizes_typographic_quotes(self):
        """CA-1.2: El sistema debe reemplazar comillas tipográficas por comillas ASCII."""
        result = normalize("\u201ctexto\u201d")
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"' in result

    def test_ca12_collapses_multiple_spaces(self):
        """CA-1.2: El sistema debe colapsar múltiples espacios en uno."""
        result = normalize("texto    con    espacios")
        assert "    " not in result

    def test_ca13_collapses_triple_newlines(self):
        """CA-1.3: El sistema debe colapsar tres o más saltos de línea a máximo dos."""
        result = normalize("párrafo 1\n\n\n\n\npárrafo 2")
        assert "\n\n\n" not in result
        assert "párrafo 1" in result
        assert "párrafo 2" in result

    def test_ca13_preserves_double_newline_between_paragraphs(self):
        """CA-1.3: El sistema debe preservar la separación entre párrafos."""
        result = normalize("párrafo 1\n\npárrafo 2")
        assert "párrafo 1\n\npárrafo 2" == result

    def test_ca21_extracts_pozo_hint(self):
        """CA-2.1: El sistema debe extraer el pozo si está en formato POZO: valor."""
        hints = extract_metadata_hints("POZO: PM-104\ncontenido técnico")
        assert hints.get("pozo") == "PM-104"

    def test_ca21_extracts_año_hint(self):
        """CA-2.1: El sistema debe extraer el año si está en formato AÑO: valor."""
        hints = extract_metadata_hints("AÑO: 2021\ncontenido técnico")
        assert hints.get("año") == "2021"

    def test_ca21_returns_empty_when_no_metadata(self):
        """CA-2.1: Si no hay metadata estructurada, debe retornar un dict vacío."""
        hints = extract_metadata_hints("texto sin estructura de metadata")
        assert hints == {}

    def test_ca21_case_insensitive_extraction(self):
        """CA-2.1: La extracción de metadata debe ser insensible a mayúsculas."""
        hints = extract_metadata_hints("pozo: LL-205")
        assert "pozo" in hints

    def test_ca31_preserves_psi(self):
        """CA-3.1: El sistema debe preservar la unidad técnica psi."""
        result = normalize("presión de 1.200 psi en el intervalo")
        assert "psi" in result

    def test_ca31_preserves_cubic_meter(self):
        """CA-3.1: El sistema debe preservar la unidad técnica m³."""
        result = normalize("caudal de 50 m³/día inyectado")
        assert "m³" in result

    def test_ca31_preserves_density_unit(self):
        """CA-3.1: El sistema debe preservar la unidad técnica g/cm³."""
        result = normalize("peso del lodo: 1.15 g/cm³")
        assert "g/cm³" in result

    def test_ca32_nonempty_input_produces_nonempty_output(self):
        """CA-3.2: Si el texto original tiene contenido, el resultado no debe quedar vacío."""
        text = "Informe de perforación del pozo PM-104, formación Quintuco."
        result = normalize(text)
        assert result != ""
        assert len(result) > 0

    def test_ca32_only_control_chars_returns_empty(self):
        """CA-3.2: Si el texto solo tiene caracteres de control, el resultado puede quedar vacío."""
        result = normalize("\x00\x01\x02")
        assert isinstance(result, str)


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
