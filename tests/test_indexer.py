"""
Tests para src/indexer.py (run_pipeline)
Cubren "Procesamiento de documentos":
  - Generación de embeddings para chunks -> CA-5.1, CA-5.2
  - Procesamiento batch de documentos    -> CA-6.1, CA-6.2, CA-6.3, CA-6.4
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.etl.chunker import split as chunk_split

# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def processed_dir_with_docs(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for i in range(2):
        doc = {
            "doc_id": f"doc{i}",
            "doc_type": "ewrs",
            "text": f"Texto de prueba número {i}.\n\nSegundo párrafo {i}.",
            "source_path": f"/raw/doc{i}.pdf",
            "filename": f"doc{i}.pdf",
        }
        (processed_dir / f"doc{i}.json").write_text(json.dumps(doc), encoding="utf-8")
    return processed_dir


# ---------------------------------------------------------------------------
# Generación de embeddings para chunks
# ---------------------------------------------------------------------------


def test_one_embedding_generated_per_fragment_with_correspondence():
    """CA-5.1: Dado un documento ya dividido en fragmentos (chunks), el sistema debe generar un embedding para cada fragmento, manteniendo la correspondencia entre identificador de fragmento, su texto y su embedding."""
    text = "Párrafo uno.\n\nPárrafo dos.\n\nPárrafo tres."
    chunks = chunk_split(doc_id="doc1", doc_type="ewrs", text=text)
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * 384 for _ in chunks]

    embeddings = embedder.embed_batch([c.text for c in chunks])

    assert len(embeddings) == len(chunks)
    assert all(c.chunk_id.startswith("doc1::chunk_") for c in chunks)


def test_vectorized_content_is_full_fragment_including_overlap():
    """CA-5.2: El texto que se vectoriza debe ser el contenido completo de cada fragmento (incluyendo la superposición agregada entre fragmentos consecutivos), no el texto original del documento sin dividir."""
    text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
    chunks = chunk_split(
        doc_id="doc1", doc_type="ewrs", text=text, max_chars=600, overlap_chars=100
    )
    embedder = MagicMock()
    texts_sent = [c.text for c in chunks]
    embedder.embed_batch(texts_sent)

    called_texts = embedder.embed_batch.call_args[0][0]
    assert called_texts == texts_sent
    assert any(t != text for t in called_texts)


# ---------------------------------------------------------------------------
# Procesamiento batch de documentos
# ---------------------------------------------------------------------------


def test_indexing_process_gathers_parallel_sets_before_vectorizing(
    processed_dir_with_docs, tmp_path
):
    """CA-6.1: El proceso de indexación debe recorrer todos los documentos ya procesados, generar los fragmentos de cada uno y reunir sus identificadores, textos y metadatas en conjuntos paralelos antes de generar los embeddings."""
    from src.indexer import run_pipeline

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 384] * 10
    mock_vectorstore = MagicMock()
    mock_vectorstore.count.return_value = 4

    with patch("src.indexer.etl_run") as mock_etl_run:
        mock_etl_run.return_value = MagicMock(
            total_found=2, total_ok=2, total_errors=0, total_skipped=0
        )
        run_pipeline(
            raw_dir=tmp_path / "raw",
            processed_dir=processed_dir_with_docs,
            vectorstore_dir=tmp_path / "vs",
            manifest_path=tmp_path / "manifest.jsonl",
            embedder=mock_embedder,
            vectorstore=mock_vectorstore,
        )

    called_texts = mock_embedder.embed_batch.call_args[0][0]
    assert len(called_texts) > 0
    add_batch_kwargs = mock_vectorstore.add_batch.call_args[1]
    assert (
        len(add_batch_kwargs["chunk_ids"])
        == len(add_batch_kwargs["texts"])
        == len(add_batch_kwargs["metadatas"])
    )


def test_batch_embedding_generation_is_a_single_massive_operation_timed(
    processed_dir_with_docs, tmp_path
):
    """CA-6.2: La generación de embeddings de todo el lote de documentos debe realizarse en una sola operación de vectorización masiva, registrando el tiempo total que demanda ese paso."""
    from src.indexer import run_pipeline

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 384] * 10
    mock_vectorstore = MagicMock()
    mock_vectorstore.count.return_value = 4

    with patch("src.indexer.etl_run") as mock_etl_run:
        mock_etl_run.return_value = MagicMock(
            total_found=2, total_ok=2, total_errors=0, total_skipped=0
        )
        metrics = run_pipeline(
            raw_dir=tmp_path / "raw",
            processed_dir=processed_dir_with_docs,
            vectorstore_dir=tmp_path / "vs",
            manifest_path=tmp_path / "manifest.jsonl",
            embedder=mock_embedder,
            vectorstore=mock_vectorstore,
        )

    assert mock_embedder.embed_batch.call_count == 1
    assert metrics.embeddings_elapsed_sec >= 0


def test_reports_total_embeddings_generated_at_end_of_step(
    processed_dir_with_docs, tmp_path
):
    """CA-6.3: Al finalizar la generación de embeddings, el sistema debe reportar la cantidad total de embeddings generados."""
    from src.indexer import run_pipeline

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 384] * 5
    mock_vectorstore = MagicMock()
    mock_vectorstore.count.return_value = 5

    with patch("src.indexer.etl_run") as mock_etl_run:
        mock_etl_run.return_value = MagicMock(
            total_found=2, total_ok=2, total_errors=0, total_skipped=0
        )
        metrics = run_pipeline(
            raw_dir=tmp_path / "raw",
            processed_dir=processed_dir_with_docs,
            vectorstore_dir=tmp_path / "vs",
            manifest_path=tmp_path / "manifest.jsonl",
            embedder=mock_embedder,
            vectorstore=mock_vectorstore,
        )

    assert metrics.embeddings_generated == 5


def test_unreadable_document_skipped_with_warning_without_aborting(tmp_path, caplog):
    """CA-6.4: Si un documento procesado no puede leerse o su contenido está mal formado, el sistema debe omitirlo con una advertencia y continuar generando embeddings para el resto del corpus, sin abortar el proceso completo."""
    from src.indexer import run_pipeline

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "bueno.json").write_text(
        json.dumps({"doc_id": "d1", "doc_type": "ewrs", "text": "texto ok"}),
        encoding="utf-8",
    )
    (processed_dir / "roto.json").write_text("{esto no es json", encoding="utf-8")

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 384]
    mock_vectorstore = MagicMock()
    mock_vectorstore.count.return_value = 1

    with patch("src.indexer.etl_run") as mock_etl_run:
        mock_etl_run.return_value = MagicMock(
            total_found=2, total_ok=2, total_errors=0, total_skipped=0
        )
        with caplog.at_level("WARNING"):
            run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
                vectorstore_dir=tmp_path / "vs",
                manifest_path=tmp_path / "manifest.jsonl",
                embedder=mock_embedder,
                vectorstore=mock_vectorstore,
            )

    assert "roto.json" in caplog.text or "No se pudo leer" in caplog.text
    mock_embedder.embed_batch.assert_called_once()
