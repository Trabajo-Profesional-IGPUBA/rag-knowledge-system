"""Tests unitarios para LLMClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import requests

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

    @patch("requests.post")
    def test_generate_stream_handles_malformed_json(self, mock_post):
        """Cubre CA-3.4 (Historia 3) — ignora JSON malformado sin propagar excepción."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            b'{"response": "ok", "done": false}',
            b'{"response": "incompleto"',  # JSON corrupto (falta cerrar la llave)
            b'{"response": "", "done": true}',
        ]
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_post.return_value = mock_resp

        client = self.client_cls(self.config_cls())

        try:
            tokens = list(client.generate_stream("prompt de prueba"))
        except Exception as e:
            assert False, f"generate_stream() no debería lanzar excepción, lanzó: {e}"

        assert tokens == ["ok"]
        
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
        mock_get.side_effect = requests.exceptions.ConnectionError("No se pudo conectar")

        client = self.client_cls(self.config_cls())

        try:
            models = client.list_models()
        except Exception as e:
            assert False, f"list_models() no debería lanzar excepción, lanzó: {e}"

        assert models == []

    @patch("requests.post")
    def test_pull_model_downloads_missing_model(self, mock_post):
        """Cubre CA-6.1 (Historia 6) — permite descargar un modelo no disponible localmente."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = self.client_cls(self.config_cls())
        client.pull_model("llama3:8b")

        assert mock_post.call_count == 1
        called_url = mock_post.call_args[0][0]
        assert called_url.endswith("/api/pull")

    @patch("requests.post")
    def test_pull_model_handles_failure_without_crashing(self, mock_post):
        """Cubre CA-6.3 (Historia 6) — informa fallo de descarga sin interrumpirse abruptamente."""
        mock_post.side_effect = requests.exceptions.ConnectionError("No se pudo conectar")

        client = self.client_cls(self.config_cls())

        try:
            result = client.pull_model("llama3:8b")
        except Exception as e:
            assert False, f"pull_model() no debería lanzar excepción, lanzó: {e}"

        assert result is False

    @patch("requests.post")
    def test_pull_model_handles_failure_without_crashing(self, mock_post):
        """Cubre CA-6.3 (Historia 6) — informa fallo de descarga sin interrumpirse abruptamente."""
        mock_post.side_effect = requests.exceptions.ConnectionError("No se pudo conectar")

        client = self.client_cls(self.config_cls())

        try:
            result = client.pull_model("llama3:8b")
        except Exception as e:
            assert False, f"pull_model() no debería lanzar excepción, lanzó: {e}"

        assert result is False
    def test_generate_reports_failure_when_server_unavailable(self):
        """Cubre CA-7.1 (Historia 7) — informa el fallo de forma clara y estructurada."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))
        resp = client.generate("test prompt")

        assert resp.ok is False

    def test_generate_error_includes_understandable_reason(self):
        """Cubre CA-7.2 (Historia 7) — el fallo incluye un motivo entendible."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))
        resp = client.generate("test prompt")

        assert resp.error is not None
        assert len(resp.error) > 0

    def test_generate_does_not_raise_on_connection_error(self):
        """Cubre CA-7.3 (Historia 7) — no se interrumpe con error técnico no controlado."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        try:
            resp = client.generate("test prompt")
        except Exception as e:
            assert False, f"generate() no debería lanzar excepción, lanzó: {e}"

        assert resp.text == ""

    def test_llmconfig_centralizes_model_parameters(self):
        """Cubre CA-8.1 (Historia 8) — centraliza los parámetros del modelo en un solo lugar."""
        config = self.config_cls(
            model="mistral:7b",
            temperature=0.5,
            top_p=0.8,
            top_k=20,
            num_predict=512,
            repeat_penalty=1.2,
        )

        assert config.model == "mistral:7b"
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.top_k == 20
        assert config.num_predict == 512
        assert config.repeat_penalty == 1.2

    def test_llmconfig_has_reasonable_defaults(self):
        """Cubre CA-8.2 (Historia 8) — valores por defecto razonables si no se especifican."""
        from src.llm.client import DEFAULT_MODEL, DEFAULT_TIMEOUT, OLLAMA_BASE_URL

        config = self.config_cls()

        assert config.model == DEFAULT_MODEL
        assert config.base_url == OLLAMA_BASE_URL
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.temperature == 0.1
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.num_predict == 1024
        assert config.repeat_penalty == 1.1

    def test_llmconfig_reusable_across_multiple_clients(self):
        """Cubre CA-8.3 (Historia 8) — la configuración se puede reutilizar en varias instancias."""
        config = self.config_cls(model="mistral:7b")

        client_a = self.client_cls(config)
        client_b = self.client_cls(config)

        assert client_a.config is config
        assert client_b.config is config
        assert client_a.config.model == client_b.config.model == "mistral:7b"

    def test_llmresponse_same_structure_success_and_failure(self):
        """Cubre CA-9.1 (Historia 9) — misma estructura sin importar éxito o fallo."""
        from src.llm.client import LLMResponse

        success_fields = set(LLMResponse(text="ok", model="llama3:8b").__dict__.keys())
        failure_fields = set(
            LLMResponse(text="", model="llama3:8b", ok=False, error="algo falló").__dict__.keys()
        )

        assert success_fields == failure_fields

    def test_llmresponse_includes_generated_text_or_empty(self):
        """Cubre CA-9.2 (Historia 9) — incluye el texto generado, o vacío si falló."""
        from src.llm.client import LLMResponse

        ok_resp = LLMResponse(text="Hola mundo", model="llama3:8b")
        fail_resp = LLMResponse(text="", model="llama3:8b", ok=False, error="error")

        assert ok_resp.text == "Hola mundo"
        assert fail_resp.text == ""

    @patch("requests.post")
    def test_llmresponse_includes_tokens_and_elapsed_time(self, mock_post):
        """Cubre CA-9.3 (Historia 9) — incluye métricas de tokens y tiempo transcurrido."""
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
        assert resp.elapsed_sec >= 0

    def test_llmresponse_indicates_success_or_failure_reason(self):
        """Cubre CA-9.4 (Historia 9) — indica éxito/fallo y el motivo en caso de falla."""
        from src.llm.client import LLMResponse

        ok_resp = LLMResponse(text="Hola mundo", model="llama3:8b")
        fail_resp = LLMResponse(text="", model="llama3:8b", ok=False, error="Ollama no disponible")

        assert ok_resp.ok is True
        assert ok_resp.error is None
        assert fail_resp.ok is False
        assert fail_resp.error == "Ollama no disponible"
           
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
