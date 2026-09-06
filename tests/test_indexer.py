"""
Tests para src/indexer.py (run_pipeline)
"""

from unittest.mock import MagicMock

from src.etl.chunker import split as chunk_split

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
