"""
Pipeline completo de indexación:
  ETL → Chunking → Embeddings → VectorStore

Orquesta todos los módulos para procesar documentos y dejarlos listos
para consulta semántica.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.etl.batch import run as etl_run
from src.etl.chunker import split as chunk_split
from src.embeddings.embedder import Embedder
from src.retrieval.vectorstore import VectorStore
from src.observability import PipelineMetrics, Timer

log = logging.getLogger(__name__)


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    vectorstore_dir: Path,
    manifest_path: Path,
    embedder: Embedder | None = None,
    vectorstore: VectorStore | None = None,
    incremental: bool = True,
) -> PipelineMetrics:
    """
    Ejecuta el pipeline completo de indexación.

    Args:
        raw_dir: directorio con PDFs originales.
        processed_dir: directorio donde se guardan JSONs procesados.
        vectorstore_dir: directorio de persistencia de ChromaDB.
        manifest_path: archivo JSONL con metadata de documentos procesados.
        embedder: instancia de Embedder (se crea si no se provee).
        vectorstore: instancia de VectorStore (se crea si no se provee).
        incremental: si True, saltea documentos ya procesados.

    Returns:
        PipelineMetrics con estadísticas del pipeline.
    """
    metrics = PipelineMetrics()

    # ── Paso 1: ETL ────────────────────────────────────────────────────────
    log.info("── Paso 1: ETL ──────────────────────────────────────────────")
    with Timer() as t:
        etl_result = etl_run(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            manifest_path=manifest_path,
            incremental=incremental,
        )
    metrics.docs_found = etl_result.total_found
    metrics.docs_ok = etl_result.total_ok
    metrics.docs_error = etl_result.total_errors
    metrics.docs_skipped = etl_result.total_skipped
    metrics.etl_elapsed_sec = t.elapsed
    log.info("ETL completado: %d OK, %d errores, %d saltados", metrics.docs_ok, metrics.docs_error, metrics.docs_skipped)

    # ── Paso 2: Chunking + Embeddings + Indexación ─────────────────────────
    log.info("── Paso 2: Chunking → Embeddings → Indexación ───────────────")

    if embedder is None:
        embedder = Embedder()

    if vectorstore is None:
        vectorstore = VectorStore(vectorstore_dir)

    # Leer JSONs procesados y generar chunks
    json_files = list(processed_dir.rglob("*.json"))
    log.info("Procesando %d documentos JSON para chunking", len(json_files))

    all_chunk_ids: list[str] = []
    all_texts: list[str] = []
    all_metadatas: list[dict] = []

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("No se pudo leer %s: %s", json_path, e)
            continue

        doc_id = data.get("doc_id", json_path.stem)
        doc_type = data.get("doc_type", "desconocido")
        text = data.get("text", "")

        chunks = chunk_split(doc_id=doc_id, doc_type=doc_type, text=text)
        metrics.chunks_generated += len(chunks)

        for chunk in chunks:
            all_chunk_ids.append(chunk.chunk_id)
            all_texts.append(chunk.text)
            all_metadatas.append({
                "doc_id": chunk.doc_id,
                "doc_type": chunk.doc_type,
                "chunk_index": chunk.chunk_index,
                "source_path": data.get("source_path", ""),
                "filename": data.get("filename", ""),
            })

    log.info("Total chunks generados: %d", metrics.chunks_generated)

    # ── Paso 3: Generar embeddings en batch ────────────────────────────────
    log.info("── Paso 3: Generando embeddings ─────────────────────────────")
    with Timer() as t:
        embeddings = embedder.embed_batch(all_texts, show_progress=True)
    metrics.embeddings_generated = len(embeddings)
    metrics.embeddings_elapsed_sec = t.elapsed

    # ── Paso 4: Indexar en vectorstore ─────────────────────────────────────
    log.info("── Paso 4: Indexando en vectorstore ─────────────────────────")
    vectorstore.add_batch(
        chunk_ids=all_chunk_ids,
        texts=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )
    metrics.vectors_indexed = vectorstore.count()

    metrics.log_summary()
    return metrics
