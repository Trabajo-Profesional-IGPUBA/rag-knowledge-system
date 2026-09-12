"""Pipeline RAG: orquesta Retriever + PromptBuilder + LLMClient para responder consultas."""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.llm.client import LLMClient, LLMResponse
from src.llm.prompt_builder import BuiltPrompt, PromptBuilder
from src.retrieval.retriever import RetrievalResult, Retriever

log = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Resultado de una consulta RAG: pregunta, respuesta, fuentes y métricas."""

    query: str
    answer: str
    retrieval: RetrievalResult
    prompt: BuiltPrompt
    llm_response: LLMResponse
    total_elapsed_sec: float = 0.0

    @property
    def ok(self) -> bool:
        """True si la generación fue exitosa y produjo una respuesta no vacía."""
        return self.llm_response.ok and bool(self.answer)

    @property
    def sources(self) -> list[dict[str, Any]]:
        """Fuentes usadas, ordenadas por relevancia."""
        return [
            {
                "filename": c["metadata"].get(
                    "filename", c["metadata"].get("doc_id", "")
                ),
                "doc_type": c["metadata"].get("doc_type", ""),
                "score": c["score"],
                "chunk_id": c["chunk_id"],
            }
            for c in self.retrieval.chunks[: self.prompt.num_chunks]
        ]

    def pretty_sources(self) -> str:
        """Devuelve las fuentes formateadas como texto legible, o "Sin fuentes" si no hay."""
        lines = []
        for i, s in enumerate(self.sources, 1):
            lines.append(
                f"  [{i}] {s['filename']} ({s['doc_type']}) — relevancia: {s['score']:.0%}"
            )
        return "\n".join(lines) if lines else "  Sin fuentes"


@dataclass
class RAGConfig:
    """Parámetros de configuración del pipeline RAG (top_k, score mínimo, streaming, filtros)."""

    top_k: int = 5
    min_score: float = 0.3
    stream: bool = False
    filters: dict[str, Any] | None = None


class RAGPipeline:
    """Orquesta Retriever + PromptBuilder + LLMClient para responder consultas RAG."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._config = config or RAGConfig()

        log.info(
            "RAGPipeline inicializado — top_k=%d, min_score=%.2f, modelo=%s",
            self._config.top_k,
            self._config.min_score,
            self._llm.config.model,
        )

    def query(self, question: str, use_history: bool = False) -> RAGResponse:
        """
        Ejecuta una consulta RAG completa.

        Args:
            question: pregunta en lenguaje natural.
            use_history: si True, incluye historial de conversación en el prompt.

        Returns:
            RAGResponse con respuesta, fuentes y métricas.
        """
        t0 = time.perf_counter()
        log.info("RAG query: '%s'", question)

        retrieval = self._retriever.retrieve(
            query=question,
            top_k=self._config.top_k,
            filters=self._config.filters,
        )

        filtered_chunks = [
            c for c in retrieval.chunks if c["score"] >= self._config.min_score
        ]

        if not filtered_chunks:
            log.warning(
                "No se encontraron chunks con score >= %.2f", self._config.min_score
            )

        built = self._prompt_builder.build(
            query=question,
            chunks=filtered_chunks,
        )

        log.info(
            "Prompt construido — %d chunks, %d chars totales",
            built.num_chunks,
            built.total_chars,
        )

        llm_resp = self._llm.generate(built.prompt)

        elapsed = time.perf_counter() - t0
        log.info("RAG completado en %.2fs", elapsed)

        return RAGResponse(
            query=question,
            answer=llm_resp.text,
            retrieval=retrieval,
            prompt=built,
            llm_response=llm_resp,
            total_elapsed_sec=elapsed,
        )

    def query_stream(self, question: str) -> Iterator[str]:
        """Ejecuta una consulta RAG en modo streaming, yieldeando tokens a medida que se generan."""
        retrieval = self._retriever.retrieve(
            query=question,
            top_k=self._config.top_k,
            filters=self._config.filters,
        )

        filtered_chunks = [
            c for c in retrieval.chunks if c["score"] >= self._config.min_score
        ]

        built = self._prompt_builder.build(query=question, chunks=filtered_chunks)

        full_response = []
        for token in self._llm.generate_stream(built.prompt):
            full_response.append(token)
            yield token
