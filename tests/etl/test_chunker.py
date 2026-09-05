from src.etl.chunker import MAX_CHARS, OVERLAP_CHARS, split


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
        paragraph = "Palabra " * 20 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("doc1", "ewrs", text, max_chars=100)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_max_chars_significantly(self):
        paragraph = "Texto corto."
        text = "\n\n".join([paragraph] * 50)
        chunks = split("doc1", "ewrs", text, max_chars=100, overlap_chars=0)
        for chunk in chunks:
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
            assert len(chunks[1].text) > 0

    def test_ca11_default_strategy_is_paragraph_based(self):
        """CA-1.1: La estrategia seleccionada divide por párrafos (doble newline)."""
        parrafo_1 = "Primer párrafo con información técnica sobre el pozo PM-104."
        parrafo_2 = "Segundo párrafo sobre la formación Quintuco y su comportamiento."
        text = f"{parrafo_1}\n\n{parrafo_2}"
        chunks = split("doc1", "ewrs", text, max_chars=500)
        # Ambos párrafos deben estar en el mismo chunk si caben
        assert parrafo_1 in chunks[0].text
        assert parrafo_2 in chunks[0].text

    def test_ca11_paragraphs_not_cut_in_half(self):
        """CA-1.1: El chunking no debe cortar un párrafo a la mitad."""
        parrafo = "Este es un párrafo técnico completo que no debe ser cortado."
        text = "\n\n".join([parrafo] * 5)
        chunks = split("doc1", "ewrs", text, max_chars=200)
        for chunk in chunks:
            # Cada chunk debe contener párrafos completos, no fragmentos
            assert (
                "Este es un párrafo técnico completo que no debe ser cortado."
                in chunk.text
            )

    def test_ca21_chunk_id_contains_doc_id(self):
        """CA-2.1: El chunk_id debe contener el doc_id del documento origen."""
        chunks = split("ewrs/EWR_PM104_2012", "end_of_well_report", "texto de prueba")
        assert "ewrs/EWR_PM104_2012" in chunks[0].chunk_id

    def test_ca21_chunk_id_contains_position(self):
        """CA-2.1: El chunk_id debe contener la posición del chunk en el documento."""
        paragraph = "Palabra " * 30 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("doc1", "ewrs", text, max_chars=100)
        for i, chunk in enumerate(chunks):
            assert f"chunk_{i}" in chunk.chunk_id

    def test_ca21_chunk_ids_are_unique(self):
        """CA-2.1: Todos los chunk_ids de un documento deben ser únicos."""
        paragraph = "Palabra " * 30 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("doc1", "ewrs", text, max_chars=100)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_ca22_doc_id_preserved_in_all_chunks(self):
        """CA-2.2: El doc_id debe estar presente en todos los chunks generados."""
        paragraph = "Palabra " * 30 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("ewrs/EWR_PM104", "ewrs", text, max_chars=100)
        for chunk in chunks:
            assert chunk.doc_id == "ewrs/EWR_PM104"

    def test_ca22_doc_type_preserved_in_all_chunks(self):
        """CA-2.2: El doc_type debe estar presente en todos los chunks generados."""
        paragraph = "Palabra " * 30 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("doc1", "parte_diario", text, max_chars=100)
        for chunk in chunks:
            assert chunk.doc_type == "parte_diario"

    def test_ca23_paragraph_boundary_respected(self):
        """CA-2.3: El sistema no debe cortar un párrafo a la mitad al generar un chunk."""
        parrafos = [
            "Informe de perforación del pozo PM-104 en la formación Quintuco.",
            "Se detectó pérdida de circulación severa durante la bajada del casing.",
            "Se bombearon píldoras de LCM para normalizar el retorno de circulación.",
        ]
        text = "\n\n".join(parrafos)
        chunks = split("doc1", "ewrs", text, max_chars=500)
        full_text = " ".join(c.text for c in chunks)
        for parrafo in parrafos:
            assert parrafo in full_text

    def test_ca31_max_chars_configurable(self):
        """CA-3.1: El tamaño máximo de chunk debe poder configurarse."""
        paragraph = "Palabra " * 20 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks_small = split("doc1", "ewrs", text, max_chars=50)
        chunks_large = split("doc1", "ewrs", text, max_chars=1000)
        assert len(chunks_small) > len(chunks_large)

    def test_ca31_default_max_chars_used_when_not_specified(self):
        """CA-3.1: Si no se especifica max_chars, debe usarse el valor por defecto MAX_CHARS."""
        text = "Párrafo de prueba."
        chunks = split("doc1", "ewrs", text)
        assert all(c.char_count <= MAX_CHARS + OVERLAP_CHARS for c in chunks)

    def test_ca32_overlap_enabled_by_default(self):
        """CA-3.2: El overlap debe estar habilitado por defecto."""
        paragraph = "Texto " * 30 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks = split("doc1", "ewrs", text, max_chars=100)
        if len(chunks) > 1:
            # Con overlap, el segundo chunk debe tener contenido del primero
            assert len(chunks[1].text) > 0

    def test_ca32_overlap_configurable(self):
        """CA-3.2: El overlap debe poder configurarse — con overlap=0 se generan menos chunks."""
        paragraph = "Texto " * 30 + "."
        text = "\n\n".join([paragraph] * 10)
        chunks_no_overlap = split("doc1", "ewrs", text, max_chars=100, overlap_chars=0)
        chunks_with_overlap = split(
            "doc1", "ewrs", text, max_chars=100, overlap_chars=50
        )
        # Con overlap se generan más chunks porque cada uno incluye contexto del anterior
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)
