from types import SimpleNamespace
from unittest.mock import MagicMock

QUERIES = [
    {
        "id": "a1",
        "query": "¿Qué pasó en PM-104?",
        "expected_keywords": ["LCM"],
        "reference_answer": "Se usó LCM a 2.450 metros.",
    },
    {
        "id": "n1",
        "query": "¿Qué pasó en ZZ-999?",
        "expected_keywords": [],
        "should_abstain": True,
    },
]
ANSWERS = {
    "¿Qué pasó en PM-104?": "Se usó LCM a 2450 metros",
    "¿Qué pasó en ZZ-999?": "No encontré información sobre ZZ-999",
}


def _result(**overrides):
    from src.llm.evaluator import ModelEvalResult

    base = {
        "model": "m",
        "query_id": "q1",
        "query": "test",
        "response": "respuesta",
        "elapsed_sec": 1.0,
        "response_length": 9,
        "keyword_hits": 0,
        "keyword_total": 0,
        "keyword_score": 0.0,
    }
    base.update(overrides)
    return ModelEvalResult(**base)


def _rag_resp(answer, context="Se usó LCM a 2.450 metros"):
    return SimpleNamespace(
        answer=answer,
        retrieval=SimpleNamespace(chunks=[{"text": context}]),
        prompt=SimpleNamespace(num_chunks=1),
    )


def _pipeline_by_query(answers, failing=()):
    pipeline = MagicMock()

    def _query(text):
        if text in failing:
            raise RuntimeError("fallo de conexión")
        return _rag_resp(answers.get(text, "respuesta"))

    pipeline.query.side_effect = _query
    return pipeline


def _mock_client(installed=("m1", "judge")):
    client = MagicMock()
    client.is_available.return_value = True
    client.list_models.return_value = list(installed)
    client.generate.return_value = (
        '{"correctness": 5, "completeness": 4, "faithfulness": 5, "abstained": false}'
    )
    return client


def _judge_client(verdict):
    client = MagicMock()
    client.generate.return_value = verdict
    return client


def _stats(**overrides):
    base = {
        "quality": 0.8,
        "avg_keyword_score": 0.7,
        "avg_semantic_similarity": 0.8,
        "p50_elapsed_sec": 5.0,
        "p95_elapsed_sec": 8.0,
        "avg_judge_faithfulness": 0.0,
        "ok_runs": 10.0,
        "hallucination_rate": 0.0,
        "false_abstention_rate": 0.0,
        "total_negative": 4.0,
    }
    base.update(overrides)
    return base
