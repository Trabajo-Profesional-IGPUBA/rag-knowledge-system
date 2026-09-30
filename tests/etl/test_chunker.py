from unittest.mock import MagicMock


from src.etl.chunker import Chunk, DoclingHybridChunker


def make_chunk(**kwargs) -> Chunk:
    """Builds a Chunk with sensible defaults for unit tests."""
    defaults = {
        "text": "Loss of circulation detected in Quintuco formation.",
        "contextualized_text": "Section: Operational Incidents\nLoss of circulation detected in Quintuco.",
        "source_file": "data/ewr/EWR_PM104_2012.pdf",
        "source_hash": "abc123",
        "page": 1,
        "chunk_index": 0,
        "wells": ["PM-104"],
    }
    defaults.update(kwargs)
    return Chunk(**defaults)


class TestChunk:

    def test_chunk_id_format(self):
        chunk = make_chunk(source_hash="abc123", chunk_index=5)
        assert chunk.chunk_id == "abc123::5"

    def test_chunk_id_unique_per_index(self):
        c1 = make_chunk(source_hash="abc123", chunk_index=0)
        c2 = make_chunk(source_hash="abc123", chunk_index=1)
        assert c1.chunk_id != c2.chunk_id

    def test_char_count(self):
        chunk = make_chunk(text="hello world")
        assert chunk.char_count == 11

    def test_chunks_never_exceed_max_tokens(self, tmp_path):
        """CA-1.2: El tamaño máximo de texto que puede procesar el modelo debe
        coincidir con el tamaño que se usa para dividir los documentos en
        fragmentos, de forma que ningún fragmento se corte sin que el sistema
        lo sepa."""
        MAX_TOKENS = 500

        # Texto largo a propósito, sin cortes naturales de párrafo antes de los
        # 500 tokens, para forzar al chunker a ejercitar el límite real.
        long_text = "Pérdida de circulación en formación Quintuco. " * 200
        sample_file = tmp_path / "sample.md"
        sample_file.write_text(f"# Sección: Incidentes Operativos\n\n{long_text}")

        chunker = DoclingHybridChunker()
        for chunk in chunker.chunk(sample_file):
            token_count = len(chunker._chunker.tokenizer.tokenizer.encode(chunk.text))
            assert token_count <= MAX_TOKENS


class TestExtractPageNo:

    def _make_raw_chunk(self, page_no: int | None) -> MagicMock:
        chunk = MagicMock()
        if page_no is not None:
            prov = MagicMock()
            prov.page_no = page_no
            doc_item = MagicMock()
            doc_item.prov = [prov]
            chunk.meta.doc_items = [doc_item]
        else:
            chunk.meta.doc_items = []
        return chunk

    def test_extracts_page_number(self):
        raw = self._make_raw_chunk(page_no=47)
        assert DoclingHybridChunker.extract_page_no(raw) == 47

    def test_returns_none_without_provenance(self):
        raw = self._make_raw_chunk(page_no=None)
        assert DoclingHybridChunker.extract_page_no(raw) is None

    def test_returns_none_with_empty_doc_items(self):
        raw = MagicMock()
        raw.meta.doc_items = []
        assert DoclingHybridChunker.extract_page_no(raw) is None
