import logging
import os
from pathlib import Path

from src.observability import setup_logging

ROOT_DIR = Path(__file__).parent
RAW_DIR = ROOT_DIR / "data" / "raw"
VECTOR_STORE_PATH = ROOT_DIR / "data" / "vectorstore"
LOG_DIR = ROOT_DIR / "logs"

DEFAULT_MAX_WORKERS = min(os.cpu_count() or 4, 4)


def _get_max_workers() -> int:
    raw = os.environ.get("INGEST_MAX_WORKERS")
    if raw is None:
        return DEFAULT_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError:
        logging.getLogger().warning(
            "INGEST_MAX_WORKERS=%r no es un número válido, usando default=%d",
            raw,
            DEFAULT_MAX_WORKERS,
        )
        return DEFAULT_MAX_WORKERS

    if value < 1:
        logging.getLogger().warning(
            "INGEST_MAX_WORKERS=%d fuera de rango, usando default=%d",
            value,
            DEFAULT_MAX_WORKERS,
        )
        return DEFAULT_MAX_WORKERS

    return value


def run():
    from src.embeddings.embedder import Embedder
    from src.etl import DoclingHybridChunker, DocumentProcessor
    from src.retrieval.vectorstore import VectorStore

    logger = logging.getLogger()

    if not RAW_DIR.exists():
        logger.error(f"No existe {RAW_DIR}")
        return

    chunker = DoclingHybridChunker()
    vector_store = VectorStore(VECTOR_STORE_PATH)
    embedder = Embedder()
    document_processor = DocumentProcessor(
        chunker=chunker, vector_repository=vector_store, embedder=embedder
    )

    try:
        for file in RAW_DIR.rglob("*.pdf"):
            document_processor.process_file(file)
    finally:
        vector_store.close()


if __name__ == "__main__":
    setup_logging(LOG_DIR)
    run()
