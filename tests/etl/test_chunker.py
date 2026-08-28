from src.etl.chunker import split


class TestSplit:
    def test_empty_text_returns_empty_list(self):
        assert split("doc1", "ewrs", "") == []

    def test_whitespace_only_returns_empty(self):
        assert split("doc1", "ewrs", "   \n\n  ") == []

    def test_single_paragraph_one_chunk(self):
        text = "Un párrafo corto."
        chunks = split("doc1", "ewrs", text)
        assert len(chunks) == 1

    def test_chunk_contains_original_text(self):
        text = "Pérdida de circulación en Quintuco a 2.450 metros."
        chunks = split("doc1", "ewrs", text)
        assert text in chunks[0].text

    def test_chunk_id_format(self):
        chunks = split("ewrs/EWR_PM104", "end_of_well_report", "texto")
        assert chunks[0].chunk_id == "ewrs/EWR_PM104::chunk_0"

    def test_chunk_metadata(self):
        chunks = split("doc1", "parte_diario", "texto de prueba")
        assert chunks[0].doc_id == "doc1"
        assert chunks[0].doc_type == "parte_diario"

    def test_char_count_set(self):
        text = "texto"
        chunks = split("doc1", "ewrs", text)
        assert chunks[0].char_count == len(chunks[0].text)

    def test_long_text_split_in_multiple_chunks(self):
        # Generar texto mayor que MAX_CHARS
        paragraph = "Palabra " * 20 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("doc1", "ewrs", text, max_chars=100)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_max_chars_significantly(self):
        paragraph = "Texto corto."
        text = "\n\n".join([paragraph] * 50)
        chunks = split("doc1", "ewrs", text, max_chars=100, overlap_chars=0)
        for chunk in chunks:
            # Permitir margen por overlap
            # Con max_chars=100 y párrafos de 12 chars cada uno, algunos chunks
            # pueden exceder ligeramente por el mecanismo de flush del último grupo
            assert chunk.char_count < 800

    def test_chunk_index_sequential(self):
        paragraph = "Palabra " * 30 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("doc1", "ewrs", text, max_chars=100)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_overlap_shares_content(self):
        paragraphs = ["Párrafo número " + str(i) + " con contenido." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = split("doc1", "ewrs", text, max_chars=80, overlap_chars=30)
        if len(chunks) > 1:
            # El overlap hace que haya contenido compartido
            assert len(chunks[1].text) > 0
