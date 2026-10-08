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
        default=["llama3.1:8b", "mistral:7b", "qwen3:8b", "llama3.2:3b", "gemma2:9b"],
        help="Nombres de modelos de Ollama a comparar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/report.json"),
        help="Path donde guardar el reporte JSON.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Modelo juez (distinto de los evaluados, idealmente más grande). "
        "Sin él no se mide corrección ni fidelidad.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Repeticiones por consulta (promedia la variación). Usá 1 para probar rápido.",
    )
    parser.add_argument(
        "--latency-penalty",
        type=float,
        default=0.05,
        help="Puntos de calidad (0-1) que se exigen por cada vez que un modelo tarda el doble "
        "que el más rápido. Más alto = se castiga más a los modelos lentos.",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=None,
        help="Fidelidad mínima (1-5) exigida; requiere --judge-model.",
    )
    args = parser.parse_args()

    if args.min_faithfulness is not None and not args.judge_model:
        parser.error("--min-faithfulness requiere --judge-model")
    if args.judge_model and args.judge_model in args.models:
        parser.error("--judge-model no puede ser uno de los modelos evaluados")
    return args


def main() -> None:
    args = parse_args()

    embedder = Embedder()
    vectorstore = VectorStore(persist_dir=args.persist_dir)

    try:
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
            embedder=embedder,
            judge_model=args.judge_model,
            n_runs=args.n_runs,
            latency_penalty=args.latency_penalty,
            min_faithfulness=args.min_faithfulness,
        )
    finally:
        vectorstore.close()


if __name__ == "__main__":
    main()
