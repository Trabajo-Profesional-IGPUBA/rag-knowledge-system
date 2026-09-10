import logging
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def run(max_workers: int | None = None):
    from src.embeddings.embedder import Embedder
    from src.etl import DoclingHybridChunker, DocumentProcessor
    from src.retrieval.vectorstore import VectorStore

    logger = logging.getLogger()

    if not RAW_DIR.exists():
        logger.error(f"No existe {RAW_DIR}")
        return

    if max_workers is None:
        max_workers = _get_max_workers()
    elif max_workers < 1:
        raise ValueError("max_workers debe ser >= 1")

    files = RAW_DIR.rglob("*.pdf")
    first_file = next(files, None)

    if first_file is None:
        logger.warning("No se encontraron archivos PDF en %s", RAW_DIR)
        return

    def file_iterator():
        yield first_file
        yield from files

    # Cantidad máxima de Futures simultáneamente en memoria.
    # Esto NO limita la RAM usada dentro de cada worker; solamente
    # evita crear un Future por cada PDF del corpus.
    max_in_flight = max_workers * 2
    chunker = DoclingHybridChunker()
    vector_store = VectorStore(VECTOR_STORE_PATH)
    embedder = Embedder()
    document_processor = DocumentProcessor(
        chunker=chunker, vector_repository=vector_store, embedder=embedder
    )

    files_iter = iter(file_iterator())
    in_flight: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_in_flight):
            file = next(files_iter, None)
            if file is None:
                break
            future = executor.submit(document_processor.process_file, file)
            in_flight[future] = file

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                file = in_flight.pop(future)
                _metrics = (
                    future.result()
                )  # el manejo de errores va en el próximo commit
                next_file = next(files_iter, None)
                if next_file is not None:
                    next_future = executor.submit(
                        document_processor.process_file, next_file
                    )
                    in_flight[next_future] = next_file


if __name__ == "__main__":
    setup_logging(LOG_DIR)
    run()
