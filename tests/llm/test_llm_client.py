"""Tests unitarios para LLMClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestLLMClient:
    def setup_method(self):
        from src.llm.client import LLMClient, LLMConfig

        self.config_cls = LLMConfig
        self.client_cls = LLMClient

    def test_is_available_false_when_no_server(self):
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))
        assert client.is_available() is False

    def test_generate_returns_error_response_on_connection_error(self):
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))
        resp = client.generate("test prompt")
        assert resp.ok is False
        assert resp.error is not None
        assert resp.text == ""

    @patch("requests.post")
    def test_generate_parses_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "Respuesta generada",
            "eval_count": 42,
            "prompt_eval_count": 100,
            "done": True,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from src.llm.client import LLMClient, LLMConfig

        client = LLMClient(LLMConfig())
        resp = client.generate("prompt de prueba")

        assert resp.ok is True
        assert resp.text == "Respuesta generada"
        assert resp.completion_tokens == 42

    @patch("requests.get")
    def test_list_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from src.llm.client import LLMClient

        client = LLMClient()
        models = client.list_models()
        assert "llama3:8b" in models
        assert "mistral:7b" in models
