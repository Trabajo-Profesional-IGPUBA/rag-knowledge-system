"""
Módulo OCR para extracción de texto en PDFs escaneados.

Estrategia:
  1. Intentar extracción digital con pdfplumber (rápido, sin pérdida).
  2. Si una página devuelve menos de MIN_CHARS caracteres, se considera
     escaneada y se aplica OCR con pytesseract sobre la imagen renderizada.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

log = logging.getLogger(__name__)

# Umbral: páginas con menos caracteres que esto se tratan como escaneadas
MIN_CHARS = 50
# DPI para renderizar páginas escaneadas
RENDER_DPI = 300
# Idioma OCR (español + inglés para terminología técnica)
OCR_LANG = "spa+eng"


def _ocr_page(page: pdfplumber.page.Page) -> str:
    """Aplica OCR a una página renderizada como imagen."""
    try:
        import pytesseract
        from PIL import Image

        img = page.to_image(resolution=RENDER_DPI).original
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        text = pytesseract.image_to_string(img, lang=OCR_LANG)
        return text.strip()
    except ImportError:
        log.warning("pytesseract o Pillow no disponibles — página escaneada omitida")
        return ""
    except Exception as exc:
        log.warning("OCR falló en página: %s", exc)
        return ""


def extract_text_from_page(page: pdfplumber.page.Page) -> tuple[str, bool]:
    """
    Extrae texto de una página.

    Returns:
        (texto, ocr_usado): texto extraído y si se usó OCR.
    """
    text = (page.extract_text() or "").strip()

    if len(text) >= MIN_CHARS:
        return text, False

    # Página escaneada o sin texto digital — usar OCR
    ocr_text = _ocr_page(page)
    return ocr_text, True


def needs_ocr(pdf_path: Path, sample_pages: int = 3) -> bool:
    """
    Detecta si un PDF requiere OCR muestreando las primeras páginas.
    Útil para logging y métricas.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = pdf.pages[:sample_pages]
            scanned = sum(
                1
                for p in pages_to_check
                if len((p.extract_text() or "").strip()) < MIN_CHARS
            )
            return scanned >= len(pages_to_check) / 2
    except Exception:
        return False
