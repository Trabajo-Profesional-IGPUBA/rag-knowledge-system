import argparse
import logging
import os
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Annotated, Any

from pypdf import PdfReader, PdfWriter

# tope del archivo, antes de cualquier otro import propio
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.observability import setup_logging

ROOT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "raw"
VECTOR_STORE_PATH = ROOT_DIR / "data" / "vectorstore"
LOG_DIR = ROOT_DIR / "logs"

DEFAULT_MAX_WORKERS = 1
DEFAULT_THEORETICAL_DATA_DIR = DEFAULT_DATA_DIR / "libros"
DEFAULT_SPLIT_THRESHOLD = 100  # páginas
DEFAULT_SPLIT_SIZE = 50  # páginas por fragmento

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
        raise argparse.ArgumentTypeError("MAX_WORKERS debe ser un entero positivo")
    if max_workers < 1:
        raise argparse.ArgumentTypeError("MAX_WORKERS debe ser un entero positivo")

    return max_workers


def existing_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_dir() and not path.is_file():
        raise argparse.ArgumentTypeError(f"'{path_str}' no existe.")
    if path.is_file() and path.suffix != ".pdf":
        raise argparse.ArgumentTypeError(f"'{path_str}' no es un archivo PDF.")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="ingesta de archivos al sistema RAG")

    parser.add_argument(
        "sources",
        nargs="*",
        type=existing_dir,
        default=[DEFAULT_DATA_DIR],
        help="Rutas a los archivos de ingesta",
    )

    parser.add_argument(
        "--max-workers",
        "-w",
        type=validate_max_workers,
        default=DEFAULT_MAX_WORKERS,
        help="Number de procesos usados en el procesamiento de SOURCES",
    )

    parser.add_argument(
        "--skip-well-extraction",
        nargs="*",
        type=existing_dir,
        default=[DEFAULT_THEORETICAL_DATA_DIR],
        metavar="PATH",
        help="Directorios o archivos donde no se extraen identificadores de pozo.",
    )

    parser.add_argument(
        "--do-ocr",
        action="store_true",
        help="Habilitar OCR para documentos escaneados (deshabilitado por defecto)",
    )

    parser.add_argument(
        "--split-threshold",
        type=int,
        default=DEFAULT_SPLIT_THRESHOLD,
        metavar="PAGES",
        help=f"Páginas máximas antes de partir el PDF (default: {DEFAULT_SPLIT_THRESHOLD})",
    )

    parser.add_argument(
        "--split-size",
        type=int,
        default=DEFAULT_SPLIT_SIZE,
        metavar="PAGES",
        help=f"Páginas por fragmento al partir (default: {DEFAULT_SPLIT_SIZE})",
    )

    args = parser.parse_args()
    return args


def file_iterator(
    sources: list[Path],
    split_threshold: int = DEFAULT_SPLIT_THRESHOLD,
    split_size: int = DEFAULT_SPLIT_SIZE,
):
    """
    Itera sobre los PDFs de las fuentes. Los PDFs que superan
    split_threshold páginas se parten en fragmentos de split_size páginas.

    Yields: (path_a_procesar, path_original)
      - path_a_procesar: el archivo real que leerá Docling (puede ser temporal)
      - path_original:   el archivo fuente para trazabilidad en los metadatos
    """
    for s in sources:
        pdfs = [s] if s.is_file() else sorted(s.rglob("*.pdf"))
        for pdf in pdfs:
            try:
                reader = PdfReader(pdf)
                n_pages = len(reader.pages)
                if n_pages <= split_threshold:
                    yield pdf, pdf
                else:
                    for inicio in range(0, n_pages, split_size):
                        writer = PdfWriter()
                        for page in reader.pages[inicio : inicio + split_size]:
                            writer.add_page(page)

                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False, prefix=f"{pdf.stem}_part_"
                        ) as tmp:
                            writer.write(tmp)

                        yield (Path(tmp.name), pdf)

            except Exception:
                # Si no se puede leer, dejarlo pasar — Docling reportará el error
                yield pdf, pdf
                continue


def run(
    sources: list[Path] = [DEFAULT_DATA_DIR],
    max_workers: PositiveInt = DEFAULT_MAX_WORKERS,
    no_well_extraction: list[Path] = [DEFAULT_THEORETICAL_DATA_DIR],
    do_ocr: bool = False,
    split_size: int = DEFAULT_SPLIT_SIZE,
    split_threshold: int = DEFAULT_SPLIT_THRESHOLD,
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
    chunker = DoclingHybridChunker(
        no_well_extraction=no_well_extraction,
        do_ocr=do_ocr,
    )
    vector_store = VectorStore(VECTOR_STORE_PATH)
    embedder = Embedder()
    write_lock = threading.Lock()
    safe_vector_store = _SerializedVectorStore(vector_store, write_lock)
    document_processor = DocumentProcessor(
        chunker=chunker,
        vector_repository=safe_vector_store,
        embedder=embedder,
    )

    docs_processed = 0
    docs_failed = 0
    chunks_processed = 0
    well_ids_extracted = 0

    files_iter = file_iterator(sources, split_threshold, split_size)
    in_flight: dict = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(max_in_flight):
                item = next(files_iter, None)
                if item is None:
                    break
                next_file, _ = item
                future = executor.submit(document_processor.process_file, next_file)
                in_flight[future] = item

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    file, _ = in_flight.pop(future)
                    try:
                        metrics = future.result()
                    except Exception:
                        docs_failed += 1
                        logger.exception("Fallo no controlado procesando %s", file.name)
                    else:
                        docs_processed += 1
                        chunks_processed += metrics.n_chunks
                        well_ids_extracted += metrics.n_well_ids_extracted
                        logger.info(
                            "[%d] OK: %s | chunks=%d | total=%.2fs",
                            docs_processed,
                            file.name,
                            metrics.n_chunks,
                            metrics.time_total_s,
                        )

                    # Se repone SIEMPRE, haya fallado o no la tarea que terminó —
                    # de lo contrario, cada falla reduce permanentemente la cantidad
                    # de tareas en vuelo, y con suficientes fallas el pool se vacía
                    # antes de agotar files_iter, dejando el resto del corpus sin
                    # procesar de forma silenciosa.
                    item = next(files_iter, None)

                    if item is not None:
                        next_file, _ = item
                        next_future = executor.submit(
                            document_processor.process_file,
                            next_file,
                        )
                        in_flight[next_future] = item

                    if (docs_processed + docs_failed) % 50 == 0:
                        logger.info(
                            "Progreso: %d procesados | %d OK | %d fallidos",
                            docs_processed + docs_failed,
                            docs_processed,
                            docs_failed,
                        )
    finally:
        vector_store.close()

    logger.info(
        "Ingesta finalizada. Procesados=%d | OK=%d | Fallidos=%d | %% Chunks con Well IDs extraidos=%.2f%%",
        docs_processed + docs_failed,
        docs_processed,
        docs_failed,
        100 * well_ids_extracted / chunks_processed,
    )


if __name__ == "__main__":
    setup_logging(LOG_DIR)
    args = parse_args()
    run(
        args.sources,
        args.max_workers,
        args.skip_well_extraction,
        args.do_ocr,
        args.split_size,
        args.split_threshold,
    )
