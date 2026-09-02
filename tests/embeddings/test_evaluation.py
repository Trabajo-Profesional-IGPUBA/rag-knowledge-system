"""
Tests para src/embeddings/evaluation.py

Cubren la historia "Pruebas de modelos candidatos":
  - Integración de modelos candidatos       -> CA-5.1, CA-5.2
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.criteria import CANDIDATE_MODELS
from src.embeddings.evaluation import (
    evaluate_candidate_model,
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


# ---------------------------------------------------------------------------
# Generación de embeddings de prueba
# CA-6.1 y CA-6.2 (extracción de textos con extract_test_texts) se testean
# en test_corpus_loader.py, ya que ese código vive en corpus_loader.py, no
# en evaluation.py. Acá solo se cubren CA-6.3 y CA-6.4.
# ---------------------------------------------------------------------------


def test_evaluate_candidate_model_calls_encode_for_warmup_and_repetitions():
    """CA-6.3: El sistema debe ejecutar una corrida de warmup no medida antes
    de las corridas de medición, para evitar que la carga inicial distorsione
    el tiempo."""
    fake_model = make_fake_model()
    with patch(
        "src.embeddings.evaluation.SentenceTransformer", return_value=fake_model
    ):
        evaluate_candidate_model(
            "modelo-x", ["texto 1", "texto 2"], EVAL_QUERIES, n_repeticiones=3
        )
    # 1 warmup + 3 repeticiones + 2 llamadas internas de measure_quality (una por query)
    warmup_y_repeticiones = 1 + 3
    assert fake_model.encode.call_count >= warmup_y_repeticiones


def test_evaluate_candidate_model_uses_median_of_repetitions(monkeypatch):
    """CA-6.4: El sistema debe calcular encode_time_sec como la mediana de
    n_repeticiones corridas de generación de embeddings."""
    fake_model = make_fake_model()
    fake_times = iter(
        [0.0, 0.10, 0.20, 0.90]
    )  # warmup + 3 corridas (una muy alta, outlier)

    def fake_perf_counter():
        try:
            return next(fake_times)
        except StopIteration:
            return 0.90

    with patch(
        "src.embeddings.evaluation.SentenceTransformer", return_value=fake_model
    ), patch(
        "src.embeddings.evaluation.time.perf_counter",
        side_effect=[
            0.0,
            0.0,  # load_time (t0, luego resta)
            0.0,
            0.10,  # repetición 1: 0.10
            0.10,
            0.30,  # repetición 2: 0.20
            0.30,
            1.20,  # repetición 3: 0.90 (outlier)
        ],
    ):
        result = evaluate_candidate_model("modelo-x", ["texto 1"], [], n_repeticiones=3)
    # La mediana de [0.10, 0.20, 0.90] es 0.20, no el promedio (0.40)
    assert result.encode_time_sec == pytest.approx(0.20, abs=0.01)
