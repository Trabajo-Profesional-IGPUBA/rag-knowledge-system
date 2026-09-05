import logging
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.etl.chunker import DoclingHybridChunker
from src.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    DocumentProcessor se encarga del procesamiento de archivos a lo largo del pipeline del rag.
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

    def process_file(self, file_path: Path):
        """
        Procesa un archivo, desde el parseo y chunk hasta el embedding y almacenamiento en la
        base de datos vectoriales.

        Args:
            - file_path: ruta al archivo a procesar
        """
        for chunk in self.chunker.chunk(file_path):
            embedded = self.embedder.embed(chunk.contextualized_text)
            self.vector_repository.add(
                chunk.chunk_id, chunk.contextualized_text, embedded, chunk.meta
            )
