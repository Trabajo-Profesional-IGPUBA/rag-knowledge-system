"""
Tests para src/embeddings/corpus_loader.py
"""

from src.embeddings.corpus_loader import (
    extract_test_texts,
)

# ---------------------------------------------------------------------------
# Generación de embeddings de prueba
# ---------------------------------------------------------------------------


def test_extract_test_texts_discards_documents_without_text():
    """CA-6.1: El sistema debe extraer los textos de prueba desde los
    documentos del corpus (extract_test_texts), descartando los que no tienen
    campo text."""
    docs = [
        {"doc_id": "1", "text": "hola"},
        {"doc_id": "2", "text": ""},
        {"doc_id": "3"},
    ]
    texts = extract_test_texts(docs)
    assert texts == ["hola"]


def test_extract_test_texts_respects_max_docs_limit():
    """CA-6.2: El sistema debe permitir limitar la cantidad de textos de
    prueba mediante el parámetro max_docs."""
    docs = [{"text": f"texto {i}"} for i in range(10)]
    texts = extract_test_texts(docs, max_docs=3)
    assert len(texts) == 3
    assert texts == ["texto 0", "texto 1", "texto 2"]


def test_extract_test_texts_returns_all_when_max_docs_not_set():
    """CA-6.2 (caso límite): sin max_docs, el límite no debe aplicarse y deben
    devolverse todos los textos disponibles."""
    docs = [{"text": f"texto {i}"} for i in range(5)]
    texts = extract_test_texts(docs)
    assert len(texts) == 5
