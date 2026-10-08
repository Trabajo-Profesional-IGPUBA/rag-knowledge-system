from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.llm.helpers import (
    ANSWERS,
    _mock_client,
    _pipeline_by_query,
)


@pytest.fixture
def mocks():
    with patch("src.llm.evaluator.RAGPipeline") as pipeline_cls, patch(
        "src.llm.evaluator.PromptBuilder"
    ), patch("src.llm.evaluator.LLMClient") as client_cls:
        client = _mock_client()
        client_cls.return_value = client
        pipeline_cls.return_value = _pipeline_by_query(ANSWERS)
        yield SimpleNamespace(
            client=client, client_cls=client_cls, pipeline_cls=pipeline_cls
        )
