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

    @patch("requests.post")
    def test_generate_returns_complete_response_in_single_call(self, mock_post):
        """Cubre CA-2.1 (Historia 2) — genera una respuesta completa de una sola vez."""
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

        client = self.client_cls(self.config_cls())
        resp = client.generate("prompt de prueba")

        assert resp.text == "Respuesta generada"
        assert mock_post.call_count == 1    

    @patch("requests.post")
    def test_generate_reports_token_usage(self, mock_post):
        """Cubre CA-2.2 (Historia 2) — informa los tokens utilizados en la generación."""
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

        client = self.client_cls(self.config_cls())
        resp = client.generate("prompt de prueba")

        assert resp.completion_tokens == 42
        assert resp.prompt_tokens == 100

    @patch("requests.post")
    def test_generate_reports_success_status(self, mock_post):
        """Cubre CA-2.3 (Historia 2) — informa si la generación fue exitosa."""
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

        client = self.client_cls(self.config_cls())
        resp = client.generate("prompt de prueba")

        assert resp.ok is True
        assert resp.error is None

    @patch("requests.post")
    def test_generate_stream_yields_tokens_progressively(self, mock_post):
        """Cubre CA-3.1 (Historia 3) — entrega la respuesta de forma progresiva, token a token."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'{"response": "Hola", "done": false}',
            b'{"response": " mundo", "done": false}',
            b'{"response": "", "done": true}',
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_post.return_value = mock_resp

        client = self.client_cls(self.config_cls())
        tokens = list(client.generate_stream("prompt de prueba"))

        assert tokens == ["Hola", " mundo"]

    @patch("requests.post")
    def test_generate_stream_stops_on_done(self, mock_post):
        """Cubre CA-3.2 (Historia 3) — detiene la entrega progresiva cuando done=True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'{"response": "Hola", "done": false}',
            b'{"response": "", "done": true}',
            b'{"response": "no deberia llegar", "done": false}',
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_post.return_value = mock_resp

        client = self.client_cls(self.config_cls())
        tokens = list(client.generate_stream("prompt de prueba"))

        assert "no deberia llegar" not in tokens
        assert tokens == ["Hola"]

    def test_generate_stream_handles_connection_error(self):
        """Cubre CA-3.3 (Historia 3) — informa error de conexión sin interrumpirse abruptamente."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        tokens = list(client.generate_stream("test prompt"))

        assert len(tokens) == 1
        assert "Error" in tokens[0]
        assert "Ollama no disponible" in tokens[0]

    def test_is_available_checks_server_before_use(self):
        """Cubre CA-4.1 (Historia 4) — permite verificar disponibilidad antes de usar el servidor."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        result = client.is_available()

        assert isinstance(result, bool)

    def test_is_available_returns_false_without_raising(self):
        """Cubre CA-4.2 (Historia 4) — informa indisponibilidad sin lanzar excepción."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        try:
            result = client.is_available()
        except Exception as e:
            assert False, f"is_available() no debería lanzar excepción, lanzó: {e}"

        assert result is False

    def test_is_available_returns_false_without_raising(self):
        """Cubre CA-4.2 (Historia 4) — informa indisponibilidad sin lanzar excepción."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        try:
            result = client.is_available()
        except Exception as e:
            assert False, f"is_available() no debería lanzar excepción, lanzó: {e}"

        assert result is False

    @patch("requests.get")
    def test_is_available_true_when_server_responds(self, mock_get):
        """Cubre CA-4.3 (Historia 4) — confirma disponibilidad cuando el servidor responde."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        client = self.client_cls(self.config_cls())
        result = client.is_available()

        assert result is True

    @patch("requests.get")
    def test_list_models_reports_available_models(self, mock_get):
        """Cubre CA-5.1 (Historia 5) — informa qué modelos están disponibles."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = self.client_cls(self.config_cls())
        models = client.list_models()

        assert models == ["llama3:8b", "mistral:7b"]

    @patch("requests.get")
    def test_list_models_returns_empty_list_on_error(self, mock_get):
        """Cubre CA-5.2 (Historia 5) — informa lista vacía en caso de error, sin fallar."""
        mock_get.side_effect = ConnectionError("No se pudo conectar")

        client = self.client_cls(self.config_cls())

        try:
            models = client.list_models()
        except Exception as e:
            assert False, f"list_models() no debería lanzar excepción, lanzó: {e}"

        assert models == []

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
