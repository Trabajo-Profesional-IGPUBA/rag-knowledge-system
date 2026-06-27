"""
Módulo de chunking: divide el texto de un documento en fragmentos indexables.

Estrategia: chunking por párrafos con ventana deslizante (overlap).
  - Divide por párrafos (doble newline).
  - Agrupa párrafos hasta alcanzar MAX_CHARS.
  - Superpone OVERLAP_CHARS del chunk anterior para preservar contexto.

Esta estrategia respeta la estructura natural de los documentos técnicos
(cada párrafo suele describir un evento o dato coherente).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tamaño máximo de cada chunk en caracteres
MAX_CHARS = 800
# Superposición entre chunks consecutivos en caracteres
OVERLAP_CHARS = 150


@dataclass
class Chunk:
    chunk_id: str          # "{doc_id}::chunk_{n}"
    doc_id: str
    doc_type: str
    text: str
    char_count: int = field(init=False)
    chunk_index: int = 0   # posición dentro del documento

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def split(
    doc_id: str,
    doc_type: str,
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """
    Divide el texto en chunks con overlap.

    Args:
        doc_id: identificador del documento origen.
        doc_type: tipo de documento (end_of_well_report, parte_diario, etc.).
        text: texto completo normalizado.
        max_chars: tamaño máximo de cada chunk.
        overlap_chars: cantidad de caracteres de overlap entre chunks.

    Returns:
        Lista de Chunk ordenados por posición en el documento.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_len = 0
    overlap_tail = ""

    def _flush(idx: int) -> None:
        nonlocal current_parts, current_len, overlap_tail
        body = "\n\n".join(current_parts)
        if overlap_tail:
            body = overlap_tail + "\n\n" + body
        body = body.strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::chunk_{idx}",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    text=body,
                    chunk_index=idx,
                )
            )
        # Calcular overlap para el siguiente chunk
        overlap_tail = body[-overlap_chars:] if len(body) > overlap_chars else body
        current_parts = []
        current_len = 0

    chunk_idx = 0
    for para in paragraphs:
        para_len = len(para)

        # Si el párrafo solo ya excede MAX_CHARS, lo partimos por frases
        if para_len > max_chars:
            if current_parts:
                _flush(chunk_idx)
                chunk_idx += 1

            # Partir por oraciones (punto + espacio)
            sentences = [s.strip() for s in para.replace(". ", ".|").split("|") if s.strip()]
            for sentence in sentences:
                if current_len + len(sentence) + 2 > max_chars and current_parts:
                    _flush(chunk_idx)
                    chunk_idx += 1
                current_parts.append(sentence)
                current_len += len(sentence) + 2
            continue

        if current_len + para_len + 2 > max_chars and current_parts:
            _flush(chunk_idx)
            chunk_idx += 1

        current_parts.append(para)
        current_len += para_len + 2

    if current_parts:
        _flush(chunk_idx)

    return chunks
