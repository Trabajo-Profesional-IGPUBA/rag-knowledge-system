"""
Tests para src/embeddings/corpus_loader.py

Cubre "ÉPICA: Evaluación de modelos de embeddings":
  - Generación de embeddings de prueba -> CA-6.1 a CA-6.2
  - Preparación y validación del corpus de evaluación: -> CA-13.1 a CA-13.5

Cubre "Cargar múltiples archivos de documentos en paralelo"
  - Carga y procesamiento de múltiples documentos -> 14.1, 14.2, 14.5, 14.6, 14.7 y 14.8

"""

import json

import pytest

from src.embeddings.corpus_loader import (
    detect_possible_duplicates,
    extract_test_texts,
    load_documents,
    summary_by_doc_type,
)


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


class TestLoadDocuments:
    # ---------------------------------------------------------------------------
    # Generación de embeddings de prueba
    # ---------------------------------------------------------------------------

    def test_extract_test_texts_discards_documents_without_text(self):
        """CA-6.1: El sistema debe extraer los textos de prueba desde los documentos del corpus,
        descartando los que no tienen campo text."""
        docs = [
            {"doc_id": "1", "text": "hola"},
            {"doc_id": "2", "text": ""},
            {"doc_id": "3"},
        ]
        texts = extract_test_texts(docs)
        assert texts == ["hola"]

    def test_extract_test_texts_respects_max_docs_limit(self):
        """CA-6.2: El sistema debe permitir limitar la cantidad de textos de prueba mediante el parámetro max_docs."""
        docs = [{"text": f"texto {i}"} for i in range(10)]
        texts = extract_test_texts(docs, max_docs=3)
        assert len(texts) == 3
        assert texts == ["texto 0", "texto 1", "texto 2"]

    def test_extract_test_texts_returns_all_when_max_docs_not_set(self):
        """CA-6.2 (caso límite): sin max_docs, el límite no debe aplicarse y deben
        devolverse todos los textos disponibles."""
        docs = [{"text": f"texto {i}"} for i in range(5)]
        texts = extract_test_texts(docs)
        assert len(texts) == 5

    # ---------------------------------------------------------------------------
    # Preparación y validación del corpus de evaluación
    # Carga de documentos del corpus
    # ---------------------------------------------------------------------------

    def test_load_documents_reads_all_jsonl_documents(self, tmp_path):
        """Carga todos los documentos JSON de un archivo JSONL."""
        jsonl = tmp_path / "processed.jsonl"
        jsonl.write_text(
            '{"doc_id": "1", "text": "hola"}\n' '{"doc_id": "2", "text": "mundo"}\n',
            encoding="utf-8",
        )

        docs = load_documents(str(jsonl))

        assert len(docs) == 2
        assert {d["doc_id"] for d in docs} == {"1", "2"}

    def test_load_documents_uses_default_data_path_when_not_specified(
        self, tmp_path, monkeypatch
    ):
        """Si no se especifica data_path, usa data/processed.jsonl."""
        monkeypatch.chdir(tmp_path)

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        (data_dir / "processed.jsonl").write_text(
            '{"doc_id": "1", "text": "hola"}\n',
            encoding="utf-8",
        )

        docs = load_documents()

        assert len(docs) == 1
        assert docs[0]["doc_id"] == "1"

    def test_load_documents_raises_when_file_does_not_exist(self, tmp_path):
        """Si el archivo especificado no existe, lanza FileNotFoundError."""
        missing_file = tmp_path / "no_existe.jsonl"

        with pytest.raises(FileNotFoundError):
            load_documents(str(missing_file))

    def test_load_documents_skips_malformed_json_without_crashing(
        self, tmp_path, capsys
    ):
        """Omite líneas JSON mal formadas sin interrumpir la carga."""
        jsonl = tmp_path / "processed.jsonl"

        jsonl.write_text(
            '{"doc_id": "1", "text": "ok"}\n'
            "{ esto no es json valido\n"
            '{"doc_id": "2", "text": "también ok"}\n',
            encoding="utf-8",
        )

        docs = load_documents(str(jsonl))

        assert len(docs) == 2
        assert {d["doc_id"] for d in docs} == {"1", "2"}

        captured = capsys.readouterr()
        assert "Line 2 is not valid JSON" in captured.out

    def test_load_documents_skips_empty_lines(self, tmp_path):
        """Ignora líneas vacías del archivo JSONL."""
        jsonl = tmp_path / "processed.jsonl"

        jsonl.write_text(
            "\n"
            '{"doc_id": "1", "text": "hola"}\n'
            "\n"
            '{"doc_id": "2", "text": "mundo"}\n'
            "\n",
            encoding="utf-8",
        )

        docs = load_documents(str(jsonl))

        assert len(docs) == 2
        assert {d["doc_id"] for d in docs} == {"1", "2"}

    # ---------------------------------------------------------------------------
    # Preparación y validación del corpus de evaluación
    # Resumen y control de calidad del corpus
    # ---------------------------------------------------------------------------

    def test_summary_by_doc_type_counts_documents_per_type(self):
        """CA-13.4: El sistema debe generar un resumen de la cantidad de documentos por cada valor de doc_type,
        usando "unknown" cuando el campo falta."""
        docs = [
            {"doc_type": "informe"},
            {"doc_type": "informe"},
            {"doc_type": "reporte"},
        ]
        summary = summary_by_doc_type(docs)
        assert summary == {"informe": 2, "reporte": 1}

    def test_summary_by_doc_type_uses_unknown_when_field_missing(self):
        """CA-13.4: El sistema debe generar un resumen de la cantidad de documentos por cada valor de doc_type,
        usando "unknown" cuando el campo falta."""
        docs = [{"doc_id": "1"}, {"doc_id": "2"}]
        summary = summary_by_doc_type(docs)
        assert summary == {"unknown": 2}

    def test_summary_by_doc_type_returns_empty_dict_for_empty_corpus(self):
        """CA-13.4 (caso límite): con un corpus vacío, el resumen debe devolver
        un diccionario vacío, sin fallar."""
        assert summary_by_doc_type([]) == {}

    def test_detect_possible_duplicates_finds_identical_text_pairs(self):
        """CA-13.5: El sistema debe detectar posibles documentos duplicados a
        partir de coincidencia exacta de texto, devolviendo los identificadores
        de los pares encontrados."""
        docs = [
            {"doc_id": "a", "text": "mismo contenido"},
            {"doc_id": "b", "text": "mismo contenido"},
            {"doc_id": "c", "text": "contenido distinto"},
        ]
        duplicates = detect_possible_duplicates(docs)
        assert duplicates == [("a", "b")]

    def test_detect_possible_duplicates_ignores_empty_text(self):
        """CA-13.5: El sistema debe detectar posibles documentos duplicados a
        partir de coincidencia exacta de texto (documentos sin texto no se
        consideran para esta detección)."""
        docs = [
            {"doc_id": "a", "text": ""},
            {"doc_id": "b", "text": "  "},
        ]
        duplicates = detect_possible_duplicates(docs)
        assert duplicates == []

    def test_detect_possible_duplicates_falls_back_to_filename_when_no_doc_id(self):
        """CA-13.5: El sistema debe detectar posibles documentos duplicados a
        partir de coincidencia exacta de texto, devolviendo los identificadores
        de los pares encontrados (usando 'filename' si falta doc_id)."""
        docs = [
            {"filename": "a.json", "text": "repetido"},
            {"filename": "b.json", "text": "repetido"},
        ]
        duplicates = detect_possible_duplicates(docs)
        assert duplicates == [("a.json", "b.json")]

    def test_detect_possible_duplicates_handles_three_or_more_identical_documents(self):
        """CA-13.5 (caso extendido): si hay más de dos documentos con el mismo
        texto, se deben reportar todos los pares consecutivos como duplicados."""
        docs = [
            {"doc_id": "a", "text": "repetido"},
            {"doc_id": "b", "text": "repetido"},
            {"doc_id": "c", "text": "repetido"},
        ]
        duplicates = detect_possible_duplicates(docs)
        assert duplicates == [("a", "b"), ("b", "c")]

    def test_loads_single_file_same_as_before(self, tmp_path):
        """CA-14.5: Cargar un único archivo (como se hace hoy) debe seguir funcionando exactamente igual que antes."""
        path = tmp_path / "processed.jsonl"
        _write_jsonl(path, [{"doc_id": "1", "text": "hola"}])

        docs = load_documents(str(path))

        assert len(docs) == 1
        assert docs[0]["doc_id"] == "1"
        assert docs[0]["_source_file"] == str(path)

    def test_raises_if_path_is_neither_file_nor_directory(self, tmp_path):
        """CA-14.6: Si el archivo indicado no existe, el sistema debe informar que no se pudo encontrar"""
        invalid_path = tmp_path / "ruta_invalida"
        with pytest.raises(FileNotFoundError):
            load_documents(str(invalid_path))

    def test_loads_all_documents_from_multiple_files(self, tmp_path):
        """CA-14.1: El sistema debe poder recibir una carpeta con varios archivos de documentos
        y devolver todos los documentos juntos en una sola lista, sin que el usuario tenga
        que indicar cada archivo uno por uno."""
        from src.embeddings.corpus_loader import load_documents_dir

        _write_jsonl(tmp_path / "a.jsonl", [{"doc_id": "1", "text": "uno"}])
        _write_jsonl(tmp_path / "b.jsonl", [{"doc_id": "2", "text": "dos"}])

        docs = load_documents_dir(str(tmp_path))

        assert len(docs) == 2
        assert {d["doc_id"] for d in docs} == {"1", "2"}

    def test_loading_is_actually_concurrent(self, tmp_path, monkeypatch):
        """CA-14.2: Cargar varios archivos debe aprovechar que se pueden leer al mismo tiempo,
        para que cargar una carpeta con muchos archivos no tarde lo mismo que sumar el tiempo de cada archivo
        por separado."""
        import threading
        import time

        from src.embeddings import corpus_loader

        for name in ["a.jsonl", "b.jsonl", "c.jsonl"]:
            _write_jsonl(tmp_path / name, [{"doc_id": name, "text": "x"}])

        concurrent_calls = []
        lock = threading.Lock()

        original = corpus_loader._load_single_file

        def _slow_load(path):
            with lock:
                concurrent_calls.append(1)
            time.sleep(0.2)  # simula E/S lenta
            with lock:
                concurrent_calls.append(-1)
            return original(path)

        monkeypatch.setattr(corpus_loader, "_load_single_file", _slow_load)

        max_simultaneous = 0
        running = 0

        def _tracking_slow_load(path):
            nonlocal running, max_simultaneous
            with lock:
                running += 1
                max_simultaneous = max(max_simultaneous, running)
            time.sleep(0.2)
            with lock:
                running -= 1
            return original(path)

        monkeypatch.setattr(corpus_loader, "_load_single_file", _tracking_slow_load)

        corpus_loader.load_documents_dir(str(tmp_path), max_workers=3)

        assert max_simultaneous > 1  # hubo más de una lectura en vuelo a la vez

    def test_returns_empty_list_when_no_matching_files(self, tmp_path):
        """CA-14.7: Si la carpeta indicada no contiene archivos que coincidan con el patrón esperado,
        el sistema debe devolver una lista vacía y no generar ningún error."""
        from src.embeddings.corpus_loader import load_documents_dir

        assert load_documents_dir(str(tmp_path)) == []

    def test_raises_if_directory_does_not_exist(self, tmp_path):
        """CA-14.8: Si la carpeta indicada no existe, el sistema debe informar
        que no se pudo encontrar mediante un error."""
        from src.embeddings.corpus_loader import load_documents_dir

        with pytest.raises(FileNotFoundError):
            load_documents_dir(str(tmp_path / "no_existe"))
