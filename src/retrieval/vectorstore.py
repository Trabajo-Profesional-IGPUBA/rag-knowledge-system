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
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)

log = logging.getLogger(__name__)

COLLECTION_NAME = "rag_knowledge"

_ID_NAMESPACE = uuid.UUID("6f8f0b1e-6b8b-4e2f-9f1e-1a2b3c4d5e6f")


def _to_point_id(chunk_id: str) -> str:
    """Convierte un chunk_id arbitrario en un UUID determinístico válido para Qdrant."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class VectorStore:
    """Wrapper sobre Qdrant con operaciones de indexación y recuperación."""

    def __init__(self, persist_dir: Path, embedding_dim: int = 384) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(persist_dir))
        self._embedding_dim = embedding_dim

        if not self._client.collection_exists(COLLECTION_NAME):
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            )

        log.info(
            "VectorStore inicializado en %s — %d documentos indexados",
            persist_dir,
            self.count(),
        )

    def count(self) -> int:
        return self._client.count(collection_name=COLLECTION_NAME).count

    # ── Indexación ────────────────────────────────────────────────────────

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Indexa un chunk individual."""
        self.add_batch(
            chunk_ids=[chunk_id],
            texts=[text],
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
        """Indexa un lote de chunks. Idempotente vía upsert."""
        if not chunk_ids:
            return

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_to_point_id(chunk_id),
                vector=embedding,
                payload={**metadata, "chunk_id": chunk_id, "text": text},
            )
            for chunk_id, embedding, metadata, text in zip(
                chunk_ids, embeddings, metadatas, texts
            )
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)
        log.info("Indexados %d chunks en vectorstore", len(chunk_ids))

    def close(self) -> None:
        """Cierra la conexión del cliente Qdrant explícitamente."""
        self._client.close()
