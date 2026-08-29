"""Tests unitarios para LLMClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestLLMClient:
    def setup_method(self):
        from src.llm.client import LLMClient, LLMConfig

        self.config_cls = LLMConfig
        self.client_cls = LLMClient

    def test_llmconfig_accepts_custom_base_url(self):
        """Cubre CA-1.1 (Historia 1) — el sistema permite configurar la URL del servidor."""
        custom_url = "http://mi-servidor-custom:9999"
        config = self.config_cls(base_url=custom_url)
        client = self.client_cls(config)

        assert client.config.base_url == custom_url

    @patch("requests.get")
    def test_uses_configured_base_url_in_requests(self, mock_get):
        """Cubre CA-1.2 (Historia 1) — el cliente usa la base_url configurada en las peticiones."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        custom_url = "http://mi-servidor-custom:9999"
        client = self.client_cls(self.config_cls(base_url=custom_url))
        client.is_available()

        called_url = mock_get.call_args[0][0]
        assert called_url == f"{custom_url}/api/tags"    

    def test_default_base_url_when_not_configured(self):
        """Cubre CA-1.3 (Historia 1) — usa una URL por defecto si no se configura ninguna."""
        from src.llm.client import OLLAMA_BASE_URL

        client = self.client_cls(self.config_cls())

        assert client.config.base_url == OLLAMA_BASE_URL
        assert client._base_url == OLLAMA_BASE_URL
        
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
