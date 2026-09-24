"""Punto de entrada para correr la evaluación comparativa de modelos LLM."""

import argparse
import logging
from pathlib import Path

from src.embeddings.embedder import Embedder
from src.llm.evaluator import evaluate_models
from src.retrieval.retriever import Retriever
from src.retrieval.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación comparativa de modelos LLM sobre el pipeline RAG."
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/vectorstore"),
        help="Carpeta donde persiste la base vectorial (Qdrant local).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.1:8b", "mistral:7b"],
        help="Nombres de modelos de Ollama a comparar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/report.json"),
        help="Path donde guardar el reporte JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedder = Embedder()
    vectorstore = VectorStore(persist_dir=args.persist_dir)

    if vectorstore.count() == 0:
        log.warning(
            "El vectorstore en '%s' está vacío. "
            "¿Es el mismo path que usaste al indexar los documentos?",
            args.persist_dir,
        )

    retriever = Retriever(embedder=embedder, vectorstore=vectorstore)

    evaluate_models(
        retriever=retriever,
        models=args.models,
        output_path=args.output,
    )

    vectorstore.close()


if __name__ == "__main__":
    main()
