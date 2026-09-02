"""
Tests para src/embeddings/corpus_loader.py
"""

import json

import pytest

from src.embeddings.corpus_loader import (
    extract_test_texts,
    load_documents,
)


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Preparación y validación del corpus de evaluación
# Carga de documentos del corpus
# ---------------------------------------------------------------------------


def test_load_documents_reads_all_json_files_recursively(tmp_path):
    """CA-13.1: El sistema debe cargar todos los documentos JSON de un
    directorio de forma recursiva, agregando la ruta de origen (_source_file)
    a cada documento cargado."""
    (tmp_path / "sub").mkdir()
    _write_json(tmp_path / "doc1.json", {"doc_id": "1", "text": "hola"})
    _write_json(tmp_path / "sub" / "doc2.json", {"doc_id": "2", "text": "mundo"})

    docs = load_documents(str(tmp_path))

    assert len(docs) == 2
    assert {d["doc_id"] for d in docs} == {"1", "2"}


def test_load_documents_adds_source_file_field(tmp_path):
    """CA-13.1: El sistema debe cargar todos los documentos JSON de un
    directorio de forma recursiva, agregando la ruta de origen (_source_file)
    a cada documento cargado."""
    _write_json(tmp_path / "doc1.json", {"doc_id": "1", "text": "hola"})

    docs = load_documents(str(tmp_path))

    assert docs[0]["_source_file"].endswith("doc1.json")


def test_load_documents_raises_when_directory_does_not_exist(tmp_path):
    """CA-13.2: Si el directorio especificado no existe, el sistema debe
    lanzar un error explícito (FileNotFoundError)."""
    missing_dir = tmp_path / "no_existe"

    with pytest.raises(FileNotFoundError):
        load_documents(str(missing_dir))
