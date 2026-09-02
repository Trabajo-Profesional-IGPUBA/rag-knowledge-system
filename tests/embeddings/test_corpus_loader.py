"""
Tests para src/embeddings/corpus_loader.py

Cubren dos tareas técnicas distintas:

"Generación de embeddings de prueba":
  - extract_test_texts()   -> CA-6.1, CA-6.2

"Preparación y validación del corpus de evaluación":
  - load_documents()             -> CA-13.1, CA-13.2, CA-13.3

"""

import json

import pytest

from src.embeddings.corpus_loader import (
    extract_test_texts,
    load_documents,
    summary_by_doc_type,
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


def test_load_documents_uses_default_data_dir_when_not_specified(tmp_path, monkeypatch):
    """CA-13.1 (valor por defecto): si no se especifica data_dir, el sistema
    debe usar 'data/processed' como directorio por defecto."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    _write_json(
        tmp_path / "data" / "processed" / "doc1.json", {"doc_id": "1", "text": "hola"}
    )

    docs = load_documents()

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "1"


def test_load_documents_raises_when_directory_does_not_exist(tmp_path):
    """CA-13.2: Si el directorio especificado no existe, el sistema debe
    lanzar un error explícito (FileNotFoundError)."""
    missing_dir = tmp_path / "no_existe"

    with pytest.raises(FileNotFoundError):
        load_documents(str(missing_dir))


def test_load_documents_skips_malformed_json_without_crashing(tmp_path, capsys):
    """CA-13.3: Si un archivo JSON está mal formado, el sistema debe omitirlo
    con una advertencia, sin interrumpir la carga del resto del corpus."""
    _write_json(tmp_path / "bueno.json", {"doc_id": "1", "text": "ok"})
    (tmp_path / "malo.json").write_text("{ esto no es json valido", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "1"
    captured = capsys.readouterr()
    assert "not valid JSON" in captured.out


# ---------------------------------------------------------------------------
# Preparación y validación del corpus de evaluación
# Resumen y control de calidad del corpus
# ---------------------------------------------------------------------------


def test_summary_by_doc_type_counts_documents_per_type():
    """CA-13.4: El sistema debe generar un resumen de la cantidad de
    documentos por cada valor de doc_type, usando "unknown" cuando el campo
    falta."""
    docs = [
        {"doc_type": "informe"},
        {"doc_type": "informe"},
        {"doc_type": "reporte"},
    ]
    summary = summary_by_doc_type(docs)
    assert summary == {"informe": 2, "reporte": 1}


def test_summary_by_doc_type_uses_unknown_when_field_missing():
    """CA-13.4: El sistema debe generar un resumen de la cantidad de
    documentos por cada valor de doc_type, usando "unknown" cuando el campo
    falta."""
    docs = [{"doc_id": "1"}, {"doc_id": "2"}]
    summary = summary_by_doc_type(docs)
    assert summary == {"unknown": 2}


def test_summary_by_doc_type_returns_empty_dict_for_empty_corpus():
    """CA-13.4 (caso límite): con un corpus vacío, el resumen debe devolver
    un diccionario vacío, sin fallar."""
    assert summary_by_doc_type([]) == {}
