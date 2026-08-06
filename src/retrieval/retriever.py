"""
Módulo de recuperación semántica (RAG retrieval).

Orquesta el flujo completo:
  consulta en texto → embedding → búsqueda en vectorstore → chunks relevantes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.embeddings.embedder import Embedder
from src.retrieval.vectorstore import VectorStore

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[dict[str, Any]]
    top_k: int

    @property
    def texts(self) -> list[str]:
        return [c["text"] for c in self.chunks]

    @property
    def best_score(self) -> float:
        return self.chunks[0]["score"] if self.chunks else 0.0


class Retriever:
    """Combina Embedder + VectorStore para recuperación semántica."""

    def __init__(self, embedder: Embedder, vectorstore: VectorStore) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Recupera los chunks más relevantes para una consulta.

        Args:
            query: pregunta en lenguaje natural.
            top_k: cantidad de chunks a recuperar.
            filters: filtros opcionales por metadata del documento.
                     Ej: {"doc_type": "parte_diario"}

        Returns:
            RetrievalResult con los chunks ordenados por relevancia.
        """
        log.info("Consulta: '%s' (top_k=%d)", query, top_k)

        query_embedding = self._embedder.embed(query)
        chunks = self._vectorstore.search(
            query_embedding=query_embedding,
            n_results=top_k,
            filters=filters,
        )

        log.info(
            "Recuperados %d chunks — mejor score: %.4f",
            len(chunks),
            chunks[0]["score"] if chunks else 0.0,
        )

        return RetrievalResult(query=query, chunks=chunks, top_k=top_k)


# ── #13 Evaluación de precisión y rendimiento ─────────────────────────────

def evaluate_topk_accuracy(
    retriever: Retriever,
    queries_with_expected: list[tuple[str, list[str]]],
    k_values: list[int] = [1, 3, 5],
) -> dict[str, float]:
    """
    Evalúa Top-K accuracy del sistema de recuperación.

    Args:
        retriever: instancia de Retriever a evaluar.
        queries_with_expected: lista de (query, [doc_ids_esperados]).
        k_values: valores de K a evaluar.

    Returns:
        Dict con accuracy para cada K. Ej: {"top_1": 0.6, "top_3": 0.8, "top_5": 1.0}
    """
    results: dict[str, list[bool]] = {f"top_{k}": [] for k in k_values}

    for query, expected_doc_ids in queries_with_expected:
        max_k = max(k_values)
        result = retriever.retrieve(query, top_k=max_k)
        retrieved_doc_ids = [c["metadata"].get("doc_id", "") for c in result.chunks]

        for k in k_values:
            top_k_ids = retrieved_doc_ids[:k]
            hit = any(exp in top_k_ids for exp in expected_doc_ids)
            results[f"top_{k}"].append(hit)

    accuracy = {
        key: round(sum(hits) / len(hits), 4) if hits else 0.0
        for key, hits in results.items()
    }

    log.info("Top-K accuracy: %s", accuracy)
    return accuracy
