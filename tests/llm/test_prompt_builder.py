"""Tests unitarios para PromptBuilder."""

import re

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

    def test_build_combines_system_context_and_query(self):
        """CA-2.1: el sistema debe construir el prompt combinando system prompt, bloque de contexto y pregunta en un único texto."""
        chunks = [self._make_chunk("dato técnico")]
        result = self.builder.build("¿cuál es la presión?", chunks)
        assert SYSTEM_PROMPT.strip() in result.prompt
        assert "dato técnico" in result.prompt
        assert "¿cuál es la presión?" in result.prompt

    def test_build_includes_chunk_text(self):
        """CA-2.1: el texto del chunk debe formar parte del bloque de contexto dentro del prompt combinado."""
        text = "Profundidad total: 2500 metros"
        chunks = [self._make_chunk(text)]
        result = self.builder.build("pregunta", chunks)
        assert text in result.prompt

    def test_context_block_is_visually_delimited(self):
        """CA-2.2: el bloque de contexto debe estar delimitado visualmente (separadores) del resto del prompt."""
        chunks = [self._make_chunk("dato técnico")]
        result = self.builder.build("pregunta", chunks)
        assert "DOCUMENTOS DE CONTEXTO RECUPERADOS" in result.prompt
        assert "═" in result.prompt

    def test_build_includes_query(self):
        """CA-2.3: el prompt final debe incluir la pregunta del usuario en una sección claramente identificada."""
        query = "¿Cuál es la presión del pozo?"
        chunks = [self._make_chunk("La presión es 3500 psi.")]
        result = self.builder.build(query, chunks)
        assert "PREGUNTA DEL USUARIO:" in result.prompt
        assert query in result.prompt

    def test_build_empty_chunks(self):
        """CA-2.4: si no hay chunks recuperados, el bloque de contexto debe contener un mensaje indicándolo en lugar de quedar vacío."""
        result = self.builder.build("pregunta", [])
        assert result.num_chunks == 0
        assert "No se encontraron documentos relevantes." in result.prompt

    def test_chunks_are_numbered_sequentially(self):
        """CA-3.1: cada chunk incluido en el contexto debe numerarse de forma secuencial."""
        chunks = [self._make_chunk(f"texto {i}", doc_id=f"doc{i}") for i in range(3)]
        result = self.builder.build("pregunta", chunks)
        assert "[1]" in result.prompt
        assert "[2]" in result.prompt
        assert "[3]" in result.prompt

    def test_chunk_includes_filename(self):
        """CA-3.2: cada chunk debe indicar el nombre del documento de origen."""
        chunks = [self._make_chunk("texto")]
        result = self.builder.build("pregunta", chunks)
        assert "PM-104_EWRS" in result.prompt

    def test_chunk_includes_doc_type(self):
        """CA-3.3: cada chunk debe indicar el tipo de documento."""
        chunks = [self._make_chunk("texto")]
        result = self.builder.build("pregunta", chunks)
        assert "end_of_well_report" in result.prompt

    def test_chunk_includes_relevance_score_as_percentage(self):
        """CA-3.4: cada chunk debe indicar el porcentaje de relevancia (score) con el que fue recuperado."""
        chunks = [self._make_chunk("texto", score=0.87)]
        result = self.builder.build("pregunta", chunks)
        assert "87%" in result.prompt

    def test_chunk_without_filename_falls_back_to_doc_id(self):
        """CA-3.5: si un chunk no tiene filename, el sistema debe usar un identificador alternativo razonable (doc_id)."""
        chunk = {
            "chunk_id": "docX::chunk_0",
            "text": "texto",
            "metadata": {"doc_id": "docX", "doc_type": "end_of_well_report"},
            "score": 0.9,
        }
        result = self.builder.build("pregunta", [chunk])
        assert "docX" in result.prompt

    def test_chunk_without_doc_type_uses_default_unknown(self):
        """CA-3.6: si un chunk no tiene doc_type, el sistema debe indicar un valor por defecto que refleje 'desconocido'."""
        chunk = {
            "chunk_id": "docY::chunk_0",
            "text": "texto",
            "metadata": {"doc_id": "docY", "filename": "docY_file"},
            "score": 0.9,
        }
        result = self.builder.build("pregunta", [chunk])
        assert "desconocido" in result.prompt

    def test_max_context_chars_is_configurable(self):
        """CA-4.1: el sistema debe permitir configurar un límite máximo de caracteres para el bloque de contexto."""
        builder = PromptBuilder(max_context_chars=200)
        assert builder._max_context_chars == 200

    def test_default_max_context_chars_is_reasonable(self):
        """CA-4.2: si no se especifica un límite, el sistema debe usar un valor por defecto razonable."""
        builder = PromptBuilder()
        assert builder._max_context_chars == 6000

    def test_build_respects_max_context(self):
        """CA-4.3: el límite configurado debe respetarse al construir el contexto, sin superarlo."""
        builder = PromptBuilder(max_context_chars=100)
        big_chunks = [self._make_chunk("x" * 200, doc_id=f"doc{i}") for i in range(5)]
        result = builder.build("pregunta", big_chunks)
        assert result.context_chars <= 100

    def test_chunks_added_in_order_until_limit(self):
        """CA-5.1: el sistema debe agregar chunks al contexto en orden hasta alcanzar el límite configurado."""
        builder = PromptBuilder(max_context_chars=5000)
        chunks = [self._make_chunk(f"texto {i}", doc_id=f"doc{i}") for i in range(3)]
        result = builder.build("pregunta", chunks)
        assert "texto 0" in result.prompt
        assert "texto 1" in result.prompt
        assert "texto 2" in result.prompt

    def test_stops_without_cutting_chunk_in_half(self):
        """CA-5.2: al llegar al límite, el sistema debe detener la incorporación de chunks sin cortar uno a la mitad."""
        builder = PromptBuilder(max_context_chars=100)
        big_chunks = [self._make_chunk("x" * 200, doc_id=f"doc{i}") for i in range(5)]
        result = builder.build("pregunta", big_chunks)
        assert re.search(r"x{1,199}(?!x)", result.prompt) is None

    def test_build_counts_chunks_correctly(self):
        """CA-5.3: el sistema debe informar cuántos chunks fueron efectivamente incluidos en el prompt final."""
        chunks = [self._make_chunk(f"texto {i}", doc_id=f"doc{i}") for i in range(3)]
        result = self.builder.build("pregunta", chunks)
        assert result.num_chunks == 3

    def test_reports_context_chars_used(self):
        """CA-5.4: el sistema debe informar la cantidad de caracteres de contexto efectivamente utilizados."""
        chunks = [self._make_chunk("texto corto")]
        result = self.builder.build("pregunta", chunks)
        assert result.context_chars > 0

    def test_no_chunks_shows_explicit_message(self):
        """CA-6.1: si no se recupera ningún chunk relevante, el contexto debe incluir un mensaje explícito indicándolo."""
        result = self.builder.build("pregunta", [])
        assert "No se encontraron documentos relevantes." in result.prompt

    def test_no_chunks_num_chunks_is_zero(self):
        """CA-6.2: si no se recupera ningún chunk relevante, el conteo de chunks incluidos en el resultado debe ser cero."""
        result = self.builder.build("pregunta", [])
        assert result.num_chunks == 0

    def test_build_with_history(self):
        """CA-7.1: el sistema debe permitir incluir el historial de la conversación dentro del prompt."""
        history = [("pregunta anterior", "respuesta anterior")]
        chunks = [self._make_chunk("texto")]
        result = self.builder.build_with_history("nueva pregunta", chunks, history)
        assert "HISTORIAL DE CONVERSACIÓN" in result.prompt
        assert "pregunta anterior" in result.prompt
        assert "respuesta anterior" in result.prompt

    def test_history_preserves_chronological_order(self):
        """CA-7.2: el historial incluido debe conservar el orden cronológico de los turnos."""
        history = [
            ("primera pregunta", "primera respuesta"),
            ("segunda pregunta", "segunda respuesta"),
        ]
        chunks = [self._make_chunk("texto")]
        result = self.builder.build_with_history("nueva pregunta", chunks, history)
        pos_primera = result.prompt.find("primera pregunta")
        pos_segunda = result.prompt.find("segunda pregunta")
        assert pos_primera < pos_segunda

    def test_history_distinguishes_user_and_assistant(self):
        """CA-7.3: cada turno del historial debe distinguir claramente qué parte corresponde al usuario y cuál al asistente."""
        history = [("pregunta anterior", "respuesta anterior")]
        chunks = [self._make_chunk("texto")]
        result = self.builder.build_with_history("nueva pregunta", chunks, history)
        assert "Usuario: pregunta anterior" in result.prompt
        assert "Asistente: respuesta anterior" in result.prompt

    def test_no_history_builds_without_errors(self):
        """CA-7.4: si no se provee historial, el prompt debe construirse igualmente sin esa sección y sin generar errores."""
        chunks = [self._make_chunk("texto")]
        result = self.builder.build_with_history("nueva pregunta", chunks, [])
        assert "nueva pregunta" in result.prompt
        assert "HISTORIAL DE CONVERSACIÓN" not in result.prompt
