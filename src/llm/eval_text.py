"""Scoring de texto: keywords, números, abstención y similitud semántica."""

import re
import unicodedata

import numpy as np

from src.embeddings.embedder import Embedder
from src.llm.eval_queries import NO_INFO_PATTERNS


def _normalize(text: str) -> str:
    """Normaliza texto: minúsculas y sin acentos, para matching más tolerante."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text


def _as_groups(keywords: list) -> list[list[str]]:
    """Convierte ['a', ['b', 'c']] en [['a'], ['b', 'c']] (grupos de sinónimos)."""
    return [[k] if isinstance(k, str) else list(k) for k in keywords]


def _score_keywords(
    text: str, keywords: list, query: str = ""
) -> tuple[int, int, float]:
    q_norm = _normalize(query)
    t_norm = _normalize(text)
    groups = [
        g for g in _as_groups(keywords) if not any(_normalize(a) in q_norm for a in g)
    ]
    hits = sum(1 for g in groups if any(_normalize(a) in t_norm for a in g))
    total = len(groups)
    return hits, total, (hits / total if total else 0.0)


def _nums(text: str) -> set[str]:
    """Números sin separadores ('2.450' == '2450'); ignora los de 1 dígito."""
    found = re.findall(r"\d+(?:[.,]\d+)*", text)
    cleaned = {n.replace(".", "").replace(",", "") for n in found}
    return {n for n in cleaned if len(n) >= 2}


def _score_numbers(text: str, reference: str | None, query: str = "") -> float | None:
    """Fracción de los números de la referencia que aparecen en `text`."""
    if not reference:
        return None
    ref = _nums(reference) - _nums(query)
    if not ref:
        return None
    return len(ref & _nums(text)) / len(ref)


def _is_no_info_response(response: str) -> bool:
    """Detecta si la respuesta es una abstención ('no encontré información...')."""
    response_norm = _normalize(response)
    return any(_normalize(p) in response_norm for p in NO_INFO_PATTERNS)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores."""
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _semantic_similarity(
    response: str, reference: str | None, embedder: Embedder | None
) -> float | None:
    """Similitud semántica (coseno) entre la respuesta generada y la referencia."""
    if not reference or embedder is None:
        return None
    resp_emb = embedder.embed(response)
    ref_emb = embedder.embed(reference)
    return round(_cosine_similarity(resp_emb, ref_emb), 4)
