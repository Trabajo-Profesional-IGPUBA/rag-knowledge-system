from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.etl.document_processor import DocumentProcessor, ProcessingMetrics

EMBEDDING_DIM = 768


def make_chunk(
    chunk_id: str = "abc::0", text: str = "some text", well: str | None = "PM-104"
):
    chunk = MagicMock()
    chunk.chunk_id = chunk_id
    chunk.text = text
    chunk.contextualized_text = f"Section: Incidents\n{text}"
    chunk.meta = {
        "well": well,
        "section": "Incidents",
        "source_file": "data/ewr/sample.pdf",
        "source_hash": "abc",
        "page": 1,
    }
    return chunk


def make_processor(chunks: list | None = None, embedding_dim: int = EMBEDDING_DIM):
    """Builds a DocumentProcessor with mocked dependencies."""
    mock_chunks = chunks if chunks is not None else [make_chunk()]
    chunker = MagicMock()
    chunker.chunk.return_value = mock_chunks

    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * embedding_dim for _ in (mock_chunks)]

    vectorstore = MagicMock()

    processor = DocumentProcessor(
        chunker=chunker,
        embedder=embedder,
        vector_repository=vectorstore,
    )
    return processor, chunker, embedder, vectorstore


class TestProcessingMetrics:

    def test_time_total_sums_stages(self):
        m = ProcessingMetrics(
            filename="sample.pdf",
            time_chunking_s=1.0,
            time_embedding_s=2.0,
            time_indexing_s=0.5,
        )
        assert m.time_total_s == pytest.approx(3.5)


class TestProcessFile:

    def test_metrics_filename(self):
        processor, *_ = make_processor()
        metrics = processor.process_file(Path("data/ewr/sample.pdf"))
        assert metrics.filename == "sample.pdf"

    def test_metrics_n_chunks(self):
        chunks = [make_chunk(f"abc::{i}") for i in range(5)]
        processor, *_ = make_processor(chunks=chunks)
        metrics = processor.process_file(Path("sample.pdf"))
        assert metrics.n_chunks == 5

    def test_chunker_called_with_file_path(self):
        processor, chunker, *_ = make_processor()
        path = Path("data/ewr/sample.pdf")
        processor.process_file(path)
        chunker.chunk.assert_called_once_with(path)

    def test_embedder_receives_contextualized_texts(self):
        chunks = [
            make_chunk("abc::0", text="chunk A"),
            make_chunk("abc::1", text="chunk B"),
        ]
        processor, _, embedder, _ = make_processor(chunks=chunks)
        processor.process_file(Path("sample.pdf"))

        expected_texts = [c.contextualized_text for c in chunks]
        embedder.embed_batch.assert_called_once_with(expected_texts)

    def test_vectorstore_receives_correct_arguments(self):
        chunks = [make_chunk("abc::0"), make_chunk("abc::1")]
        processor, _, embedder, vectorstore = make_processor(chunks=chunks)

        fake_embeddings = [[0.1] * EMBEDDING_DIM, [0.2] * EMBEDDING_DIM]
        embedder.embed_batch.return_value = fake_embeddings

        processor.process_file(Path("sample.pdf"))

        vectorstore.add_batch.assert_called_once_with(
            chunk_ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
            embeddings=fake_embeddings,
            metadatas=[c.meta for c in chunks],
        )

    def test_embedding_uses_contextualized_not_plain_text(self):
        """Embedder must receive contextualized_text, not plain text."""
        chunk = make_chunk(text="plain text")
        chunk.contextualized_text = "Section: Header\nplain text"

        processor, _, embedder, _ = make_processor(chunks=[chunk])
        processor.process_file(Path("sample.pdf"))

        called_with = embedder.embed_batch.call_args[0][0]
        assert called_with == ["Section: Header\nplain text"]

    def test_vectorstore_stores_plain_text_not_contextualized(self):
        """VectorStore documents field must contain plain text for display."""
        chunk = make_chunk(text="plain text")
        chunk.contextualized_text = "Section: Header\nplain text"

        processor, _, _, vectorstore = make_processor(chunks=[chunk])
        processor.process_file(Path("sample.pdf"))

        call_kwargs = vectorstore.add_batch.call_args.kwargs
        assert call_kwargs["texts"] == ["plain text"]


class TestProcessFileEmpty:

    def test_skips_embedding_when_no_chunks(self):
        processor, _, embedder, vectorstore = make_processor(chunks=[])
        processor.process_file(Path("empty.pdf"))
        embedder.embed_batch.assert_not_called()
        vectorstore.add_batch.assert_not_called()

    def test_returns_zero_chunks_metric(self):
        processor, *_ = make_processor(chunks=[])
        metrics = processor.process_file(Path("empty.pdf"))
        assert metrics.n_chunks == 0

    def test_returns_zero_times_when_no_chunks(self):
        processor, *_ = make_processor(chunks=[])
        metrics = processor.process_file(Path("empty.pdf"))
        assert metrics.time_embedding_s == 0.0
        assert metrics.time_indexing_s == 0.0
