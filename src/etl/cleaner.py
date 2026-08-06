"""
Módulo de limpieza y normalización de texto extraído de documentos técnicos.

Operaciones aplicadas:
  - Colapsar múltiples saltos de línea
  - Normalizar espacios en blanco
  - Eliminar caracteres de control y artefactos de OCR comunes
  - Normalizar guiones y comillas
  - Preservar unidades técnicas (psi, m³, g/cm³, etc.)
"""

from __future__ import annotations

import re
import unicodedata

# ── Patrones de limpieza ───────────────────────────────────────────────────
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Artefactos comunes de OCR
_OCR_ARTIFACTS = re.compile(r"[|]{2,}|[_]{3,}|\.{4,}")
# Guiones largos → guion estándar
_DASHES = re.compile(r"[–—]")
# Comillas tipográficas → ASCII
_QUOTES = re.compile(r'[“”„""„]')
_SINGLE_QUOTES = re.compile(r"[''`]")


def normalize(text: str) -> str:
    """
    Normaliza texto crudo extraído de un PDF.
    Preserva estructura de párrafos (doble newline).
    """
    if not text:
        return ""

    # 1. Normalización Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Eliminar caracteres de control
    text = _CONTROL_CHARS.sub("", text)

    # 3. Normalizar guiones y comillas
    text = _DASHES.sub("-", text)
    text = _QUOTES.sub('"', text)
    text = _SINGLE_QUOTES.sub("'", text)

    # 4. Eliminar artefactos de OCR
    text = _OCR_ARTIFACTS.sub("", text)

    # 5. Normalizar espacios en blanco en línea (no saltos)
    lines = []
    for line in text.splitlines():
        line = _MULTI_SPACE.sub(" ", line).strip()
        lines.append(line)
    text = "\n".join(lines)

    # 6. Colapsar múltiples líneas vacías → máximo dos
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def extract_metadata_hints(text: str) -> dict[str, str]:
    """
    Intenta extraer hints de metadata del texto normalizado.
    Busca patrones como 'POZO: PM-104', 'AÑO: 2021', etc.
    """
    hints: dict[str, str] = {}

    patterns = {
        "pozo": re.compile(r"(?:POZO|Pozo)[:\s]+([A-Z]{2,4}-\d{2,4})", re.I),
        "año": re.compile(r"(?:AÑO|Año)[:\s]+(\d{4})", re.I),
        "tipo": re.compile(r"(?:TIPO|Tipo)[:\s]+(.+?)(?:\n|$)", re.I),
        "seccion": re.compile(r"(?:SECCIÓN|SECCION|Sección)[:\s]+(.+?)(?:\n|$)", re.I),
    }

    for key, pattern in patterns.items():
        m = pattern.search(text)
        if m:
            hints[key] = m.group(1).strip()

    return hints
