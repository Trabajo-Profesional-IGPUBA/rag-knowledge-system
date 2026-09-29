import re
from dataclasses import dataclass
from typing import Self

_PATTERNS: dict[str, re.Pattern] = {
    "well": re.compile(
        r"(?:pozo\s*[:\-–]?\s*)?"  # prefijo opcional
        r"([A-Z]{2,4})"  # letras (CH, LL, YPF...)
        r"[\s\-]?"  # separador opcional (guión o espacio)
        r"(\d{1,4}[A-Z]?)"  # número
        r"(?=\s|$|[.,;)])",  # lookahead: fin de identificador
        re.IGNORECASE,
    ),
    "section": re.compile(r"(?:seccion|sección)[:\s]+(.+?)(?:\n|$)", re.IGNORECASE),
}


@dataclass
class ChunkMeta:
    well: str | None
    section: str | None

    @classmethod
    def from_text(cls, text: str) -> Self:
        well_m = _PATTERNS["well"].search(text)
        if well_m is not None:
            well = _normalize_well(well_m.group(1), well_m.group(2))
        else:
            well = None

        section_m = _PATTERNS["section"].search(text)
        section = section_m.group(1).strip() if section_m is not None else None

        return cls(
            well=well,
            section=section,
        )


def _normalize_well(letters: str, digits: str) -> str:
    """Convierte las partes del identificador al formato canónico LETRAS-NÚMERO."""
    return f"{letters.upper()}-{digits.upper()}"
