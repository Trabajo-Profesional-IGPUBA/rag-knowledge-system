import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.etl.chunker import DoclingHybridChunker
from src.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Silenciar warnings internos de Docling sobre recuperación de celdas PDF mediante fallback.
# Este comportamiento es esperado en documentos con tablas complejas y no afecta el resultado.
warnings.filterwarnings(
    "ignore",
    message=".*Orphan pdf_cell.*",
    category=UserWarning,
)


@dataclass
class ProcessingMetrics:
    """Métricas recolectadas durante el procesamiento de un archivo."""

    filename: str
    n_chunks: int = 0
    time_chunking_s: float = 0.0
    time_embedding_s: float = 0.0
    time_indexing_s: float = 0.0

    @property
    def time_total_s(self) -> float:
        return self.time_chunking_s + self.time_embedding_s + self.time_indexing_s

    def log(self) -> None:
        logger.info(
            "[%s] chunks=%d | chunking=%.2fs | embedding=%.2fs | "
            "indexing=%.2fs | total=%.2fs",
            self.filename,
            self.n_chunks,
            self.time_chunking_s,
            self.time_embedding_s,
            self.time_indexing_s,
            self.time_total_s,
        )


class DocumentProcessor:
    """
    DocumentProcessor se encarga del procesamiento de archivos a lo largo
    del pipeline del RAG.
    """

    def __init__(
        self,
        chunker: DoclingHybridChunker,
        embedder: Embedder,
        vector_repository: VectorStore,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_repository = vector_repository

    def process_file(self, file_path: Path) -> ProcessingMetrics:
        """
        Procesa un archivo desde el parseo y chunking hasta el embedding
        y almacenamiento en la base de datos vectorial.

        Args:
            file_path: ruta al archivo a procesar.

        Returns:
            ProcessingMetrics con los tiempos y cantidad de chunks generados.
        """
        metrics = ProcessingMetrics(filename=file_path.name)
        logger.info("Processing: %s", file_path.name)

        t0 = time.perf_counter()
        chunks = list(self.chunker.chunk(file_path))
        metrics.time_chunking_s = time.perf_counter() - t0
        metrics.n_chunks = len(chunks)

        if not chunks:
            logger.warning("[%s] No chunks generated — skipping", file_path.name)
            return metrics

        t0 = time.perf_counter()
        embeddings = self.embedder.embed_batch([c.contextualized_text for c in chunks])
        metrics.time_embedding_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.vector_repository.add_batch(
            chunk_ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.meta for c in chunks],
        )
        metrics.time_indexing_s = time.perf_counter() - t0

        metrics.log()
        return metrics
