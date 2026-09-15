"""Construcción de prompts para el sistema RAG de IGPUBA, combinando system prompt, contexto recuperado e historial."""

from dataclasses import dataclass
from typing import Any

# Instrucciones base que definen el rol, el idioma y las reglas de uso del contexto para el asistente técnico.
SYSTEM_PROMPT = """Sos un asistente técnico especializado en documentación de pozos petroleros del IGPUBA.

Tu función es responder preguntas técnicas basándote EXCLUSIVAMENTE en los documentos de contexto proporcionados.

Reglas:
1. Respondé siempre en español.
2. Basate ÚNICAMENTE en la información del contexto. No inventes datos ni uses conocimiento externo.
3. Si la respuesta no está en el contexto, decí claramente: "No encontré información sobre esto en los documentos disponibles."
4. Cuando des datos técnicos (presiones, profundidades, fechas, producciones), citá el documento fuente entre corchetes: [Fuente: nombre_documento].
5. Sé preciso y conciso. Evitá repetir información del contexto textualmente.
6. Si hay información contradictoria entre documentos, mencionalo.
"""

# Plantilla que ensambla el prompt final: system prompt + bloque de contexto + pregunta del usuario.
_PROMPT_TEMPLATE = """{system_prompt}

═══════════════════════════════════════
DOCUMENTOS DE CONTEXTO RECUPERADOS:
═══════════════════════════════════════
{context_block}
═══════════════════════════════════════

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""

# Plantilla para formatear cada chunk individual junto con su metadata (documento, tipo, relevancia).
_CHUNK_TEMPLATE = """[{idx}] Documento: {filename} | Tipo: {doc_type} | Relevancia: {score:.0%}
{text}
"""


@dataclass
class BuiltPrompt:
    """Resultado de construir un prompt: el texto final junto con metadatos sobre chunks y longitudes."""

    prompt: str
    query: str
    num_chunks: int
    context_chars: int
    total_chars: int


class PromptBuilder:
    """Arma prompts para el LLM combinando la query del usuario con chunks de contexto recuperados, respetando un límite de caracteres."""

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        max_context_chars: int = 6000,
    ) -> None:
        """Guarda el system prompt y el límite máximo de caracteres permitidos para el bloque de contexto."""
        self._system_prompt = system_prompt
        self._max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> BuiltPrompt:
        """Construye el prompt final incorporando chunks uno a uno hasta llegar al límite de caracteres, sin cortar ninguno a la mitad."""
        context_parts: list[str] = []
        context_chars = 0

        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", meta.get("doc_id", f"doc_{idx}"))
            doc_type = meta.get("doc_type", "desconocido")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")

            chunk_str = _CHUNK_TEMPLATE.format(
                idx=idx,
                filename=filename,
                doc_type=doc_type,
                score=score,
                text=text,
            )

            if context_chars + len(chunk_str) > self._max_context_chars:
                break

            context_parts.append(chunk_str)
            context_chars += len(chunk_str)

        context_block = (
            "\n".join(context_parts)
            if context_parts
            else "No se encontraron documentos relevantes."
        )

        prompt = _PROMPT_TEMPLATE.format(
            system_prompt=self._system_prompt,
            context_block=context_block,
            query=query,
        )

        return BuiltPrompt(
            prompt=prompt,
            query=query,
            num_chunks=len(context_parts),
            context_chars=context_chars,
            total_chars=len(prompt),
        )

    def build_with_history(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        history: list[tuple[str, str]],
        max_history_turns: int = 3,
    ) -> BuiltPrompt:
        """Construye el prompt igual que build(), insertando además las últimas max_history_turns interacciones previas antes de la pregunta."""
        history_str = ""
        if history:
            recent = history[-max_history_turns:]
            lines = []
            for q, a in recent:
                lines.append(f"Usuario: {q}")
                lines.append(f"Asistente: {a}")
            history_str = "\nHISTORIAL DE CONVERSACIÓN:\n" + "\n".join(lines) + "\n"

        built = self.build(query, chunks)
        prompt_with_history = built.prompt.replace(
            "PREGUNTA DEL USUARIO:",
            history_str + "PREGUNTA DEL USUARIO:",
        )

        return BuiltPrompt(
            prompt=prompt_with_history,
            query=query,
            num_chunks=built.num_chunks,
            context_chars=built.context_chars,
            total_chars=len(prompt_with_history),
        )
