"""
Tests para src/embeddings/evaluation.py

Cubren la historia "Pruebas de modelos candidatos":
  - Integración de modelos candidatos       -> CA-5.1, CA-5.2
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.embeddings.criteria import CANDIDATE_MODELS
from src.embeddings.evaluation import (
    evaluate_models,
)

# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    {
        "query": "q1",
        "doc_correcto": "correcto1",
        "doc_incorrecto": "incorrecto1",
    },
    {
        "query": "q2",
        "doc_correcto": "correcto2",
        "doc_incorrecto": "incorrecto2",
    },
]


def make_fake_model(dim=8, accuracy_correct_wins=True):
    """
    Crea un mock de SentenceTransformer cuyo .encode() devuelve vectores
    controlados: si accuracy_correct_wins=True, el doc_correcto queda más
    cerca de la query que el doc_incorrecto (para simular buena calidad).
    """
    model = MagicMock()

    def fake_encode(texts, convert_to_numpy=True):
        return np.array([_text_to_vector(t, dim, accuracy_correct_wins) for t in texts])

    model.encode.side_effect = fake_encode
    return model


def _text_to_vector(text, dim, accuracy_correct_wins):
    """
    La query siempre apunta al eje +1. El doc 'ganador' (el que debe quedar
    más similar a la query) también apunta a +1; el 'perdedor' apunta a -1.
    accuracy_correct_wins decide si el ganador es el doc correcto o el incorrecto.
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    base = rng.random(dim)

    if text.startswith("incorrecto"):
        gana = not accuracy_correct_wins
    elif text.startswith("correcto"):
        gana = accuracy_correct_wins
    else:
        gana = True  # la query siempre está del lado "ganador" por definición

    direction = 1.0 if gana else -1.0
    return np.ones(dim) * 0.9 * direction + base * 0.01


# ---------------------------------------------------------------------------
# Integración de modelos candidatos
# ---------------------------------------------------------------------------


def test_evaluate_models_uses_candidate_models_list_by_default():
    """CA-5.1: El sistema debe mantener una lista de modelos candidatos
    (CANDIDATE_MODELS) con nombre y nota descriptiva."""
    fake_model = make_fake_model()
    with patch(
        "src.embeddings.evaluation.SentenceTransformer", return_value=fake_model
    ):
        results = evaluate_models(
            test_texts=["texto 1", "texto 2"],
            evaluation_queries=EVAL_QUERIES,
            results_path=None,
        )
    expected_names = {c["name"] for c in CANDIDATE_MODELS}
    assert set(results.keys()) == expected_names


def test_evaluate_models_respects_explicit_candidates_list():
    """CA-5.1 (extensión): la lista de modelos candidatos debe poder acotarse
    explícitamente sin perder la definición por defecto de CANDIDATE_MODELS."""
    fake_model = make_fake_model()
    with patch(
        "src.embeddings.evaluation.SentenceTransformer", return_value=fake_model
    ):
        results = evaluate_models(
            test_texts=["texto 1"],
            evaluation_queries=EVAL_QUERIES,
            candidates=["modelo-x"],
            results_path=None,
        )
    assert set(results.keys()) == {"modelo-x"}


def test_english_only_model_not_present_in_candidate_models():
    """CA-5.2: Los modelos que no soportan español (ej. modelos solo-inglés
    como all-MiniLM-L6-v2) deben quedar excluidos de la lista, con la
    exclusión documentada."""
    candidate_names = {c["name"] for c in CANDIDATE_MODELS}
    assert "all-MiniLM-L6-v2" not in candidate_names
