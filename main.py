from pathlib import Path

from src.observability import setup_logging

ROOT_DIR = Path(__file__).parent
RAW_DIR = ROOT_DIR / "data" / "raw"
VECTOR_STORE_PATH = ROOT_DIR / "data" / "vectorstore"
LOG_DIR = ROOT_DIR / "logs"


def run():
    import logging

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
        chunker=chunker,
        vector_repository=vector_store,
        embedder=embedder,
    )

    for file in RAW_DIR.rglob("*.pdf"):
        document_processor.process_file(file)


if __name__ == "__main__":
    setup_logging(LOG_DIR)

    run()
