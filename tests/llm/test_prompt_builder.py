"""Tests unitarios para PromptBuilder."""

from src.llm.prompt_builder import SYSTEM_PROMPT, PromptBuilder


class TestPromptBuilder:
    def setup_method(self):

        self.builder = PromptBuilder(max_context_chars=5000)

    def _make_chunk(self, text: str, score: float = 0.9, doc_id: str = "doc1") -> dict:
        return {
            "chunk_id": f"{doc_id}::chunk_0",
            "text": text,
            "metadata": {
                "doc_id": doc_id,
                "doc_type": "end_of_well_report",
                "filename": "PM-104_EWRS",
            },
            "score": score,
        }

    def test_build_includes_query(self):
        query = "¿Cuál es la presión del pozo?"
        chunks = [self._make_chunk("La presión es 3500 psi.")]
        result = self.builder.build(query, chunks)
        assert query in result.prompt

    def test_build_includes_chunk_text(self):
        text = "Profundidad total: 2500 metros"
        chunks = [self._make_chunk(text)]
        result = self.builder.build("pregunta", chunks)
        assert text in result.prompt

    def test_build_empty_chunks(self):
        result = self.builder.build("pregunta", [])
        assert result.num_chunks == 0
        assert "No se encontraron documentos" in result.prompt

    def test_build_respects_max_context(self):

        builder = PromptBuilder(max_context_chars=100)
        big_chunks = [self._make_chunk("x" * 200, doc_id=f"doc{i}") for i in range(5)]
        result = builder.build("pregunta", big_chunks)
        assert result.context_chars <= 200

    def test_build_counts_chunks_correctly(self):
        chunks = [self._make_chunk(f"texto {i}", doc_id=f"doc{i}") for i in range(3)]
        result = self.builder.build("pregunta", chunks)
        assert result.num_chunks == 3

    def test_build_with_history(self):
        history = [("pregunta anterior", "respuesta anterior")]
        chunks = [self._make_chunk("texto")]
        result = self.builder.build_with_history("nueva pregunta", chunks, history)
        assert "pregunta anterior" in result.prompt
        assert "respuesta anterior" in result.prompt

    def test_system_prompt_defines_technical_role(self):
        """CA-1.1: el system prompt debe establecer el rol técnico especializado (IGPUBA)."""
        assert "asistente técnico" in SYSTEM_PROMPT
        assert "IGPUBA" in SYSTEM_PROMPT

    def test_system_prompt_restricts_to_context(self):
        """CA-1.2: el system prompt debe indicar que se basa exclusivamente en el contexto."""
        assert "EXCLUSIVAMENTE" in SYSTEM_PROMPT
        assert "conocimiento externo" in SYSTEM_PROMPT

    def test_system_prompt_requires_spanish(self):
        """CA-1.3: el system prompt debe indicar que se responde siempre en español."""
        assert "Respondé siempre en español" in SYSTEM_PROMPT

    def test_system_prompt_handles_missing_info(self):
        """CA-1.4: el system prompt debe indicar qué decir cuando falta información en el contexto."""
        assert (
            "No encontré información sobre esto en los documentos disponibles"
            in SYSTEM_PROMPT
        )

    def test_system_prompt_requires_source_citation(self):
        """CA-1.5: el system prompt debe exigir citar la fuente de datos técnicos entre corchetes."""
        assert "[Fuente: nombre_documento]" in SYSTEM_PROMPT

    def test_system_prompt_handles_contradictions(self):
        """CA-1.6: el system prompt debe indicar que se mencionen las contradicciones entre documentos."""
        assert "contradictoria" in SYSTEM_PROMPT
