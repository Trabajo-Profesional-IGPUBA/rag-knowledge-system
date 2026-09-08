"""
Módulo de base de datos vectorial.

Motor seleccionado: Qdrant
  - Corre 100% local sin servidor externo (QdrantClient(path=...)).
  - Persiste automáticamente en disco.
  - Soporte nativo de filtros por metadata (doc_type, doc_id, chunk_index)
    vía Filter/FieldCondition.
  - Mismo código funciona en modo local o contra un servidor Qdrant real
    (Docker o Qdrant Cloud) cambiando únicamente la inicialización del
    cliente (path= vs host/port o url=).
  - Los embeddings se generan externamente (componente Embedder, basado en
    sentence-transformers) y se reciben ya calculados en add()/add_batch().
    Se optó por este diseño para desacoplar la generación de embeddings
    del motor de almacenamiento: permite migrar de backend vectorial
    (como ya ocurrió de Chroma a Qdrant) sin reescribir ni regenerar el
    corpus, y facilita testear el store con vectores fijos/determinísticos.
    Este módulo no usa el soporte de embebido automático que trae
    qdrant-client (FastEmbed, basado en ONNX Runtime, no en
    sentence-transformers) ni ninguna integración directa con esa librería.

Motor anterior: ChromaDB
  - Persistía en modo local usando SQLite.
  - Este módulo tampoco usaba su soporte de embebido automático
    (EmbeddingFunction): los embeddings ya llegaban calculados.
  - Se migra a Qdrant sin cambiar la interfaz pública de esta clase, por
    lo que el resto del pipeline no requiere modificaciones.

Alternativas evaluadas:
  - FAISS: no incluye persistencia en disco ni filtrado por metadata
    como funcionalidad propia; descartada.
  - Weaviate: overhead innecesario para esta etapa.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    VectorParams,
)

log = logging.getLogger(__name__)

COLLECTION_NAME = "rag_knowledge"

_ID_NAMESPACE = uuid.UUID("6f8f0b1e-6b8b-4e2f-9f1e-1a2b3c4d5e6f")


def _to_point_id(chunk_id: str) -> str:
    """Convierte un chunk_id arbitrario en un UUID determinístico válido para Qdrant."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
    """
    Traduce el formato de filtros usado en el dominio (dict simple, con
    soporte de {"$in": [...]}"} a un Filter nativo de Qdrant.

    Ej: {"doc_type": "parte_diario"}
    Ej: {"doc_type": {"$in": ["ewrs", "workover_report"]}}
    """
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        if isinstance(value, dict) and "$in" in value:
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value["$in"])))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=conditions)


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

    # ── Recuperación semántica ───────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recupera los chunks más relevantes por similitud semántica.
        Devuelve: chunk_id, text, metadata, distance, score.
        """
        total = self.count()
        if total == 0:
            return []

        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=min(n_results, total),
            query_filter=_build_filter(filters),
            with_payload=True,
        ).points

        hits = []
        for point in results:
            payload = dict(point.payload or {})
            chunk_id = payload.pop("chunk_id", str(point.id))
            text = payload.pop("text", "")
            score = round(point.score, 4)
            distance = round(1 - point.score, 4)
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": payload,
                    "distance": distance,
                    "score": score,
                }
            )
        return hits

    # ── Borrado y reinicio ───────────────────────────────────────────────

    def delete_by_doc(self, doc_id: str) -> None:
        """Elimina todos los chunks de un documento."""
        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )
        log.info("Chunks eliminados para doc_id=%s", doc_id)

    def reset(self) -> None:
        """Elimina y recrea la colección. Útil para reprocesar todo."""
        self._client.delete_collection(COLLECTION_NAME)
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self._embedding_dim, distance=Distance.COSINE
            ),
        )
        log.info("VectorStore reiniciado")

    def close(self) -> None:
        """Cierra la conexión del cliente Qdrant explícitamente."""
        self._client.close()
