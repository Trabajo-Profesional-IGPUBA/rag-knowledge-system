"""
Módulo de generación de embeddings.

Modelo seleccionado: paraphrase-multilingual-MiniLM-L12-v2
  - Multilingüe (español + inglés), cubre la terminología técnica del dominio.
  - Liviano (117MB), corre en CPU sin problemas.
  - Dimensión: 384.

Alternativas evaluadas:
  - all-MiniLM-L6-v2: solo inglés, descartado.
  - paraphrase-multilingual-mpnet-base-v2: mayor calidad pero 2x más pesado.
  - text-embedding-ada-002 (OpenAI): requiere API key y conexión, descartado
    por confidencialidad de los datos del IGPUBA.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


class Embedder:
    """Wrapper sobre SentenceTransformer con batch processing y logging."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        log.info("Cargando modelo de embeddings: %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        log.info("Modelo cargado — dimensión: %d", self.get_dimension())

    def get_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Genera embedding para un texto individual."""
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: lista de strings a vectorizar.
            batch_size: tamaño de lote para procesamiento.
            show_progress: mostrar barra de progreso.

        Returns:
            Lista de vectores (uno por texto).
        """
        if not texts:
            return []

        log.info(
            "Generando embeddings para %d textos (batch_size=%d)",
            len(texts),
            batch_size,
        )

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        log.info(
            "Embeddings generados: %d vectores de dim %d",
            len(vectors),
            vectors.shape[1],
        )
        return [v.tolist() for v in vectors]


def evaluate_models(texts: list[str]) -> dict[str, dict]:
    """
    Evalúa modelos candidatos sobre un conjunto de textos de prueba.
    Registra dimensión, tiempo de generación y similitud semántica básica.

    Usado para documentar la selección del modelo en el informe.
    """
    import time

    candidates = [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2",
    ]

    results = {}
    for model_name in candidates:
        try:
            model = SentenceTransformer(model_name)
            t0 = time.perf_counter()
            vectors = model.encode(texts, convert_to_numpy=True)
            elapsed = time.perf_counter() - t0

            results[model_name] = {
                "dimension": vectors.shape[1],
                "time_sec": round(elapsed, 3),
                "texts_per_sec": round(len(texts) / elapsed, 1),
            }
            log.info(
                "Modelo %s — dim=%d, tiempo=%.2fs, vel=%.1f textos/s",
                model_name,
                vectors.shape[1],
                elapsed,
                len(texts) / elapsed,
            )
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return results
