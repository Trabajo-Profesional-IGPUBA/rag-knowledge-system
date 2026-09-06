"""
Módulo de base de datos vectorial.

Motor seleccionado: ChromaDB (modo embebido)
  - Corre 100% local sin Docker ni servidor externo.
  - Persiste automáticamente con SQLite.
  - Soporte nativo de filtros por metadata (doc_type, doc_id, chunk_index).
  - Integración directa con sentence-transformers.

Alternativas evaluadas (ver épica #9):
  - FAISS: sin persistencia nativa ni filtros de metadata, descartado.
  - Qdrant: mejor para producción con filtros complejos, candidato futuro.
  - Weaviate: requiere Docker, overhead innecesario para esta etapa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)

COLLECTION_NAME = "rag_knowledge"


class VectorStore:
    """Wrapper sobre ChromaDB con operaciones de indexación y recuperación."""

    def __init__(self, persist_dir: Path) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "VectorStore inicializado en %s — %d documentos indexados",
            persist_dir,
            self._collection.count(),
        )

    # ── Indexación ────────────────────────────────────────────────────────

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Indexa un chunk individual."""
        self._collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def add_batch(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Indexa un lote de chunks.
        Usa upsert para ser idempotente (reprocesar no duplica).
        """
        if not chunk_ids:
            return
        lengths = {
            "chunk_ids": len(chunk_ids),
            "texts": len(texts),
            "embeddings": len(embeddings),
            "metadatas": len(metadatas),
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"add_batch: chunk_ids, texts, embeddings y metadatas deben "
                f"tener la misma longitud. Recibido: {lengths}"
            )
        self._collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        log.info("Indexados %d chunks en vectorstore", len(chunk_ids))

    def count(self) -> int:
        return self._collection.count()

    # ── Recuperación semántica (#12) ──────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recupera los chunks más relevantes por similitud semántica.

        Args:
            query_embedding: vector de la consulta.
            n_results: cantidad de resultados a retornar (Top-K).
            filters: filtros opcionales por metadata.
                     Ej: {"doc_type": "parte_diario"}
                     Ej: {"doc_type": {"$in": ["ewrs", "workover_report"]}}

        Returns:
            Lista de dicts con keys: chunk_id, text, metadata, distance.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)

        hits = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": meta,
                    "distance": round(dist, 4),
                    "score": round(1 - dist, 4),  # similitud coseno
                }
            )

        return hits

    def delete_by_doc(self, doc_id: str) -> None:
        """Elimina todos los chunks de un documento."""
        self._collection.delete(where={"doc_id": doc_id})
        log.info("Chunks eliminados para doc_id=%s", doc_id)

    def reset(self) -> None:
        """Elimina y recrea la colección. Útil para reprocesar todo."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("VectorStore reiniciado")
