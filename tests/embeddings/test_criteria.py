"""
Tests para src/embeddings/criteria.py
"""

from src.embeddings.criteria import (
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
