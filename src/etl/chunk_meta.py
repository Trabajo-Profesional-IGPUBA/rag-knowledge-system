import re

# Se prioriza recall sobre precision: un chunk puede contener
# múltiples identificadores y el patrón puede producir falsos positivos.
# Esto es preferible a perder menciones de pozos relevantes para retrieval.
_WELL_PATTERN = re.compile(
    r"([A-Z]{2,4})" r"[\s\-]?" r"(\d{1,4}[A-Z]?)" r"(?=\s|$|[.,;)])",
    re.IGNORECASE,
)


def _normalize_well(letters: str, digits: str) -> str:
    """Convierte las partes del identificador al formato canónico LETRAS-NÚMERO."""
    return f"{letters.upper()}-{digits.upper()}"


def extract_wells(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            _normalize_well(match.group(1), match.group(2))
            for match in _WELL_PATTERN.finditer(text)
        )
    )
