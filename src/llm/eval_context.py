"""Información recuperada por el buscador."""

from typing import Any

CONTEXT_OK_THRESHOLD = 0.5
_CHUNK_TEXT_KEYS = ("text", "content", "document", "chunk_text", "page_content")


def _extract_context(rag_resp: Any) -> str:
    """Texto de los chunks que realmente entraron al prompt del LLM."""
    chunks = rag_resp.retrieval.chunks[: rag_resp.prompt.num_chunks]
    parts = []
    for c in chunks:
        text = next((c[k] for k in _CHUNK_TEXT_KEYS if c.get(k)), None)
        if text:
            parts.append(str(text))
    if not parts:
        raise RuntimeError(
            f"No encontré el texto del chunk. Claves disponibles: {list(chunks[0].keys()) if chunks else 'sin chunks'}"
        )
    return "\n---\n".join(parts)
