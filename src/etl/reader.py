import re
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

from .models import PageData, ProcessedDoc

DOC_TYPE_MAP: dict[str, str] = {
    "ewrs":           "end_of_well_report",
    "partes_diarios": "parte_diario",
    "workovers":      "workover_report",
    "estimulaciones": "estimulacion",
    "fallas_bes":     "falla_bes",
    "reservorio":     "reservorio",
}

_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    return _MULTI_NEWLINE.sub("\n\n", text.strip())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_skeleton(pdf_path: Path, raw_dir: Path, ts: str) -> ProcessedDoc:
    """Construye un ProcessedDoc vacío con la metadata de ruta."""
    rel      = pdf_path.relative_to(raw_dir)
    folder   = rel.parts[0]
    doc_type = DOC_TYPE_MAP.get(folder, "desconocido")
    doc_id   = str(rel.with_suffix(""))

    return ProcessedDoc(
        doc_id       = doc_id,
        source_path  = str(rel),
        doc_type     = doc_type,
        filename     = pdf_path.stem,
        page_count   = 0,
        char_count   = 0,
        text         = "",
        pages        = [],
        extracted_at = ts,
        error        = None,
    )


def extract(pdf_path: Path, raw_dir: Path) -> ProcessedDoc:
    ts  = _now_iso()
    doc = _make_skeleton(pdf_path, raw_dir, ts)

    try:
        pages_data: list[PageData] = []
        chunks: list[str]          = []

        with pdfplumber.open(pdf_path) as pdf:
            doc.page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                raw   = page.extract_text() or ""
                clean = _clean(raw)
                pages_data.append(PageData(page_num=i, text=clean))
                chunks.append(clean)

        doc.pages      = pages_data
        doc.text       = "\n\n".join(chunks)
        doc.char_count = len(doc.text)

    except Exception as exc:
        doc.error = f"{type(exc).__name__}: {exc}"

    return doc
