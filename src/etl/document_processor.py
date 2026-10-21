import logging
import time
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import transformers
from transformers import AutoTokenizer

from src.embeddings.embedder import DEFAULT_MODEL, Embedder
from src.etl.chunker import DoclingHybridChunker
from src.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Silenciar warnings internos de transformers y Docling que no afectan el resultado.
transformers.logging.set_verbosity_error()
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("MatchingPostProcessor").setLevel(logging.ERROR)

# Tokenizer para truncar el texto contextualizado antes de embeddear.
# El modelo tiene un límite de 512 tokens; el texto contextualizado puede superarlo.
_TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
_MAX_TOKENS = 500
_CHUNK_BATCH_SIZE = 16


def _truncate(text: str) -> str:
    """Trunca el texto a _MAX_TOKENS tokens solo si es necesario."""
    token_ids = _TOKENIZER.encode(text)
    if len(token_ids) <= _MAX_TOKENS:
        return text
    logger.warning(
        "Texto contextualizado truncado: %d → %d tokens",
        len(token_ids),
        _MAX_TOKENS,
    )
    truncated = _TOKENIZER.encode(text, truncation=True, max_length=_MAX_TOKENS)
    return _TOKENIZER.decode(truncated, skip_special_tokens=True)


@dataclass
class ProcessingMetrics:
    """Métricas recolectadas durante el procesamiento de un archivo."""

    filename: str
    n_chunks: int = 0
    n_well_ids_extracted: int = 0
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

    def process_file(
        self, file_path: Path, source_file: Path | None = None
    ) -> ProcessingMetrics:
        """
        Procesa un archivo desde el parseo y chunking hasta el embedding
        y almacenamiento en la base de datos vectorial.

        Args:
            file_path: ruta al archivo a procesar.
            source_file: nombre original de archivos grandes cuando se lo recibe partido.

        Returns:
            ProcessingMetrics con los tiempos y cantidad de chunks generados.
        """
        metrics = ProcessingMetrics(filename=file_path.name)
        logger.info("Processing: %s", file_path.name)

        chunks_iter = self.chunker.chunk(file_path, source_file)

        if not chunks_iter:
            logger.warning("[%s] No chunks generated — skipping", file_path.name)
            return metrics

        t0 = time.perf_counter()
        for chunk_batch in batched(chunks_iter, _CHUNK_BATCH_SIZE):
            metrics.time_chunking_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            embeddings = self.embedder.embed_batch(
                [_truncate(c.contextualized_text) for c in chunk_batch]
            )
            metrics.time_embedding_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            self.vector_repository.add_batch(
                chunk_ids=[c.chunk_id for c in chunk_batch],
                texts=[c.text for c in chunk_batch],
                embeddings=embeddings,
                metadatas=[c.meta for c in chunk_batch],
            )
            metrics.time_indexing_s += time.perf_counter() - t0
            metrics.n_chunks += len(chunk_batch)
            metrics.n_well_ids_extracted += sum(
                c.wells is not None for c in chunk_batch
            )
            t0 = time.perf_counter()

        metrics.log()

        if source_file is not None and file_path != source_file:
            file_path.unlink(missing_ok=True)

        return metrics
