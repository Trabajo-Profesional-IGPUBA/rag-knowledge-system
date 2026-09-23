from src.etl.cleaner import normalize


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

    def test_preserves_psi_pressure_unit(self):
        text = "presión de inyección: 1.200 psi en el intervalo Quintuco"
        assert "psi" in normalize(text)

    def test_preserves_kpa_pressure_unit(self):
        text = "gradiente de presión de 9.800 kPa/m en la formación"
        assert "kPa" in normalize(text)

    def test_preserves_bar_pressure_unit(self):
        text = "presión de cierre instantáneo (ISIP): 350 bar"
        assert "bar" in normalize(text)

    def test_preserves_m3_dia_flow_unit(self):
        text = "caudal de inyección: 50 m³/día promedio durante el piloto"
        assert "m³/día" in normalize(text)

    def test_preserves_bbl_volume_unit(self):
        text = "se bombearon 80 bbl de salmuera pesada por el espacio anular"
        assert "bbl" in normalize(text)

    def test_preserves_m3_h_flow_unit(self):
        text = "pérdida de circulación severa de 15 m³/h en la formación"
        assert "m³/h" in normalize(text)

    def test_preserves_g_cm3_density_unit(self):
        text = "peso del lodo reducido a un máximo de 1.15 g/cm³"
        assert "g/cm³" in normalize(text)

    def test_preserves_lb_gal_density_unit(self):
        text = "densidad del fluido de control: 9.2 lb/gal"
        assert "lb/gal" in normalize(text)

    def test_preserves_lcm_abbreviation(self):
        text = "se bombearon dos píldoras de LCM de alta concentración"
        assert "LCM" in normalize(text)

    def test_preserves_bes_abbreviation(self):
        text = "falla del sistema BES por baja aislación eléctrica"
        assert "BES" in normalize(text)

    def test_preserves_ewr_abbreviation(self):
        text = "informe final de perforación EWR del pozo PM-104"
        assert "EWR" in normalize(text)

    def test_preserves_ocr_abbreviation(self):
        text = "extracción de texto mediante OCR avanzado"
        assert "OCR" in normalize(text)

    def test_preserves_pm104_well_identifier(self):
        text = "intervención de workover en el pozo PM-104 del bloque central"
        assert "PM-104" in normalize(text)

    def test_preserves_ll205_well_identifier(self):
        text = "rotura de varillas a 1.820 metros en el pozo LL-205"
        assert "LL-205" in normalize(text)

    def test_preserves_ch45_well_identifier(self):
        text = "screen-out prematuro durante estimulación hidráulica en CH-45"
        assert "CH-45" in normalize(text)

    def test_preserves_timestamps_in_daily_report(self):
        text = "08:00 - Se constata pozo parado por rotura de varillas.\n09:30 - Se procede a ahogar el pozo con 40 bbl."
        result = normalize(text)
        assert "08:00" in result
        assert "09:30" in result