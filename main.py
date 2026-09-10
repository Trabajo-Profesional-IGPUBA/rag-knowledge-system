import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

# tope del archivo, antes de cualquier otro import propio
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.observability import setup_logging

ROOT_DIR = Path(__file__).parent
RAW_DIR = ROOT_DIR / "data" / "raw"
VECTOR_STORE_PATH = ROOT_DIR / "data" / "vectorstore"
LOG_DIR = ROOT_DIR / "logs"

DEFAULT_MAX_WORKERS = min(os.cpu_count() or 4, 4)


class _SerializedVectorStore:
    """
    Envuelve add y add_batch con un lock; delega el resto sin cambios.

    La documentación de qdrant-client en modo local (QdrantClient
    (path=...)) no garantiza explícitamente que sea seguro escribir
    concurrentemente desde múltiples threads sobre la misma instancia.
    Se serializa por precaución.

    Se protege también add() aunque el pipeline actual solo llame a
    add_batch(): add() delega internamente en self.add_batch() del
    objeto envuelto (self._inner), NO en el add_batch() de este
    wrapper, así que si en algún momento algo llama a add() directo,
    bypassearía el lock si no estuviera cubierto acá también.

    """

    def __init__(self, inner: Any, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def add(self, *args, **kwargs):
        with self._lock:
            return self._inner.add(*args, **kwargs)

    def add_batch(self, *args, **kwargs):
        with self._lock:
            return self._inner.add_batch(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


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
    write_lock = threading.Lock()
    safe_vector_store = _SerializedVectorStore(vector_store, write_lock)
    document_processor = DocumentProcessor(
        chunker=chunker,
        vector_repository=safe_vector_store,
        embedder=embedder,
    )

    processed = 0
    failed = 0

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
                try:
                    metrics = future.result()
                except Exception:
                    failed += 1
                    logger.exception("Fallo no controlado procesando %s", file.name)
                else:
                    processed += 1
                    logger.info(
                        "[%d] OK: %s | chunks=%d | total=%.2fs",
                        processed,
                        file.name,
                        metrics.n_chunks,
                        metrics.time_total_s,
                    )

                    # Reponemos exactamente una tarea por cada
                    # tarea terminada.
                    next_file = next(files_iter, None)

                    if next_file is not None:
                        next_future = executor.submit(
                            document_processor.process_file,
                            next_file,
                        )
                        in_flight[next_future] = next_file

                    if (processed + failed) % 50 == 0:
                        logger.info(
                            "Progreso: %d procesados | %d OK | %d fallidos",
                            processed + failed,
                            processed,
                            failed,
                        )
    logger.info(
        "Ingesta finalizada. Procesados=%d | OK=%d | Fallidos=%d",
        processed + failed,
        processed,
        failed,
    )


if __name__ == "__main__":
    setup_logging(LOG_DIR)
    run()
