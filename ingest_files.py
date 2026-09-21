import argparse
import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Annotated, Any

# tope del archivo, antes de cualquier otro import propio
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.observability import setup_logging

ROOT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "raw"
VECTOR_STORE_PATH = ROOT_DIR / "data" / "vectorstore"
LOG_DIR = ROOT_DIR / "logs"

DEFAULT_MAX_WORKERS = min(os.cpu_count() or 4, 4)
type PositiveInt = Annotated[int, "must be > 0"]


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


def validate_max_workers(max_workers_raw: str) -> int:
    try:
        max_workers = int(max_workers_raw)
    except ValueError:
        raise argparse.ArgumentTypeError("MAX_WORKERS must be a positive integer")
    if max_workers < 1:
        raise argparse.ArgumentTypeError("MAX_WORKERS must be a positive integer")

    return max_workers


def existing_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_dir() and not path.is_file():
        raise argparse.ArgumentTypeError(f"'{path_str}' does not exist.")
    if path.is_file() and path.suffix != ".pdf":
        raise argparse.ArgumentTypeError(f"'{path_str}' is not a PDF file.")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest local documents to RAG")

    parser.add_argument(
        "sources",
        nargs="*",
        type=existing_dir,
        default=[
            str(DEFAULT_DATA_DIR)  # Pasado como string para validarlo con existing_dir
        ],
        help="Path to raw files",
    )

    parser.add_argument(
        "--max-workers",
        "-w",
        type=validate_max_workers,
        default=DEFAULT_MAX_WORKERS,
        help="Number of procceses",
    )

    args = parser.parse_args()
    return args


def file_iterator(sources):
    for s in sources:
        if s.is_file():
            yield s
        else:
            yield from s.rglob("*.pdf")


def run(
    sources: list[Path] = [DEFAULT_DATA_DIR],
    max_workers: PositiveInt = DEFAULT_MAX_WORKERS,
):
    from src.embeddings.embedder import Embedder
    from src.etl import DoclingHybridChunker, DocumentProcessor
    from src.retrieval.vectorstore import VectorStore

    logger = logging.getLogger()

    if next(file_iterator(sources), None) is None:
        logger.warning("No se encontraron archivos PDF")
        return

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

    files_iter = file_iterator(sources)
    in_flight: dict = {}
    try:
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
    finally:
        vector_store.close()

    logger.info(
        "Ingesta finalizada. Procesados=%d | OK=%d | Fallidos=%d",
        processed + failed,
        processed,
        failed,
    )


if __name__ == "__main__":
    setup_logging(LOG_DIR)
    args = parse_args()
    run(args.sources, args.max_workers)
