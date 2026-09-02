"""
Tests para src/embeddings/criteria.py

Cubren la historia "Definición de criterios de evaluación":
  - Identificación de métricas de calidad       -> CA-1.1, CA-1.2, CA-1.3
  - Identificación de métricas de rendimiento    -> CA-2.1, CA-2.2, CA-2.3

"""

from src.embeddings.criteria import (
    PERFORMANCE_METRICS,
    QUALITY_METRICS,
)

# ---------------------------------------------------------------------------
# Identificación de métricas de calidad
# ---------------------------------------------------------------------------


def test_quality_metrics_define_retrieval_accuracy():
    """CA-1.1: El sistema debe definir `retrieval_accuracy` como la proporción de
    consultas donde la similitud coseno al documento correcto supera a la del
    incorrecto (acierto top-1)."""
    assert "retrieval_accuracy" in QUALITY_METRICS
    desc = QUALITY_METRICS["retrieval_accuracy"].lower()
    assert "top-1" in desc or "top 1" in desc
    assert "similitud" in desc


def test_quality_metrics_define_avg_margin():
    """CA-1.2: El sistema debe definir `avg_margin` como la diferencia promedio
    entre la similitud del documento correcto y la del incorrecto."""
    assert "avg_margin" in QUALITY_METRICS
    desc = QUALITY_METRICS["avg_margin"].lower()
    assert "diferencia" in desc
    assert "promedio" in desc


def test_quality_metrics_define_avg_similarity_correct():
    """CA-1.3: El sistema debe definir `avg_similarity_correct` como la similitud
    promedio al documento correcto, para detectar modelos que compriman el
    espacio vectorial (similitud alta indiscriminadamente)."""
    assert "avg_similarity_correct" in QUALITY_METRICS
    desc = QUALITY_METRICS["avg_similarity_correct"].lower()
    assert "comprima" in desc or "compresión" in desc or "irrelevante" in desc


def test_quality_metrics_has_exactly_three_metrics():
    """Test auxiliar (no corresponde a un CA textual propio): verifica que las
    tres métricas de calidad de CA-1.1, CA-1.2 y CA-1.3 sean exactamente las
    que están definidas, sin faltantes ni agregados sin documentar."""
    assert set(QUALITY_METRICS.keys()) == {
        "retrieval_accuracy",
        "avg_margin",
        "avg_similarity_correct",
    }


# ---------------------------------------------------------------------------
# Identificación de métricas de rendimiento
# ---------------------------------------------------------------------------


def test_performance_metrics_define_load_and_encode_time():
    """CA-2.1: El sistema debe definir `load_time_sec` como el tiempo de carga
    del modelo en memoria."""
    assert "load_time_sec" in PERFORMANCE_METRICS
    assert "encode_time_sec" in PERFORMANCE_METRICS


def test_performance_metrics_define_throughput():
    """CA-2.2: El sistema debe definir `encode_time_sec` y `texts_per_sec` como
    tiempo total y throughput de generación de embeddings."""
    assert "texts_per_sec" in PERFORMANCE_METRICS
    assert "textos" in PERFORMANCE_METRICS["texts_per_sec"].lower()


def test_performance_metrics_define_embedding_dim():
    """CA-2.3: El sistema debe definir `embedding_dim` como la dimensión del
    vector resultante."""
    assert "embedding_dim" in PERFORMANCE_METRICS
    assert "dimensión" in PERFORMANCE_METRICS["embedding_dim"].lower()
