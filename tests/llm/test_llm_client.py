"""
Cubre "ÉPICA: Cliente LLM sobre Ollama":
  - Conexión a Ollama vía API REST local configurable por URL-> CA-1.1 Y  CA-1.3
  - Generación de respuesta completa en modo no streaming-> CA-2.1 a CA-2.3
  - Generación de respuesta en modo streaming token a token-> CA-3.1 a CA-3.4
  - Health check de disponibilidad del servidor antes de procesar-> CA-4.1 a CA-4.3
  - Listado de modelos disponibles en la instancia de Ollama-> CA-5.1 a CA-5.2
  - Descarga automática de modelo si no está presente localmente-> CA-6.1 a CA-6.3
  - Manejo de error de conexión devolviendo LLMResponse con ok=False-> CA-7.1 a CA-7.3
  - LLMConfig como dataclass centralizado de parámetros del modelo-> CA-8.1 a CA-8.3
  - LLMResponse como dataclass de resultado con texto, tokens y tiempo-> CA-9.1 a CA-9.4

Cubre "El asistente puede dar respuestas distintas cada vez, aunque la pregunta y los documentos sean los mismos":
  - Uso de la configuración con el número fijado-> CA-11.1
  - Uso de la configuración sin el número fijado-> CA-12.1

"""

from unittest.mock import MagicMock, patch

import requests


class TestLLMClient:
    def setup_method(self):
        from src.llm.client import LLMClient, LLMConfig

        self.config_cls = LLMConfig
        self.client_cls = LLMClient

    def test_llmconfig_accepts_custom_base_url(self):
        """CA-1.1: El sistema debe permitir configurar la dirección del servidor de Ollama,
        en lugar de tener una dirección fija."""
        custom_url = "http://mi-servidor-custom:9999"
        config = self.config_cls(base_url=custom_url)
        client = self.client_cls(config)

        assert client.config.base_url == custom_url

    @patch("requests.get")
    def test_uses_configured_base_url_in_requests(self, mock_get):
        """CA-1.2: El sistema debe usar la dirección configurada para todas las comunicaciones con el servidor."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        custom_url = "http://mi-servidor-custom:9999"
        client = self.client_cls(self.config_cls(base_url=custom_url))
        client.is_available()

        called_url = mock_get.call_args[0][0]
        assert called_url == f"{custom_url}/api/tags"

    def test_default_base_url_when_not_configured(self):
        """CA-1.3: Si no se configura ninguna dirección, el sistema debe usar una dirección por defecto razonable."""
        from src.llm.client import OLLAMA_BASE_URL

        client = self.client_cls(self.config_cls())

        assert client.config.base_url == OLLAMA_BASE_URL
        assert client._base_url == OLLAMA_BASE_URL

    @patch("requests.post")
    def test_generate_returns_complete_response_in_single_call(self, mock_post):
        """CA-2.1: El sistema debe poder generar una respuesta completa a partir de una consulta,
        entregándola de una sola vez cuando está lista."""
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
        """CA-2.2: Junto con la respuesta, el sistema debe informar cuántos recursos (tokens)
        se utilizaron en la generación."""
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
        """CA-2.3: El sistema debe informar si la generación fue exitosa o no."""
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
        """CA-3.1: El sistema debe poder entregar la respuesta generada de forma progresiva,
        a medida que se va produciendo, en lugar de esperar a que esté completa."""
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
        """CA-3.2: El sistema debe detener la entrega progresiva cuando la generación finaliza."""
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
        """CA-3.3: Si ocurre un problema de conexión durante la entrega progresiva,
        el sistema debe informarlo sin interrumpirse de forma abrupta."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        tokens = list(client.generate_stream("test prompt"))

        assert len(tokens) == 1
        assert "Error" in tokens[0]
        assert "Ollama no disponible" in tokens[0]

    @patch("requests.post")
    def test_generate_stream_handles_malformed_json(self, mock_post):
        """CA-3.4: Si llega un fragmento de respuesta dañado o incompleto mientras el asistente está respondiendo,
        el sistema no debe cortarse ni romperse, tiene que ignorar ese fragmento puntual
        y seguir mostrando el resto de la respuesta con normalidad."""
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
        """CA-4.1: El sistema debe poder verificar si el servidor está disponible antes de intentar usarlo."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        result = client.is_available()

        assert isinstance(result, bool)

    def test_is_available_returns_false_without_raising(self):
        """CA-4.2: Si el servidor no está disponible, la verificación debe informarlo sin interrumpir
        la ejecución con un error técnico."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        try:
            result = client.is_available()
        except Exception as e:
            assert False, f"is_available() no debería lanzar excepción, lanzó: {e}"

        assert result is False

    @patch("requests.get")
    def test_is_available_true_when_server_responds(self, mock_get):
        """CA-4.3: Si el servidor está disponible, la verificación debe confirmarlo."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        client = self.client_cls(self.config_cls())
        result = client.is_available()

        assert result is True

    @patch("requests.get")
    def test_list_models_reports_available_models(self, mock_get):
        """CA-5.1: El sistema debe poder informar qué modelos están disponibles en el servidor."""
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
        """CA-5.2: Si no se puede obtener la lista (por error o falta de conexión),
        el sistema debe informar una lista vacía en lugar de fallar."""
        mock_get.side_effect = requests.exceptions.ConnectionError(
            "No se pudo conectar"
        )

        client = self.client_cls(self.config_cls())

        try:
            models = client.list_models()
        except Exception as e:
            assert False, f"list_models() no debería lanzar excepción, lanzó: {e}"

        assert models == []

    @patch("requests.post")
    def test_pull_model_downloads_missing_model(self, mock_post):
        """CA-6.1: El sistema debe poder descargar un modelo que no está disponible localmente."""
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
        """CA-6.3: Si la descarga falla, el sistema debe informarlo sin interrumpirse de forma abrupta."""
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "No se pudo conectar"
        )

        client = self.client_cls(self.config_cls())

        try:
            result = client.pull_model("llama3:8b")
        except Exception as e:
            assert False, f"pull_model() no debería lanzar excepción, lanzó: {e}"

        assert result is False

    def test_generate_reports_failure_when_server_unavailable(self):
        """CA-7.1: Cuando el servidor no está disponible,
        el sistema debe informar el fallo de forma clara y estructurada."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))
        resp = client.generate("test prompt")

        assert resp.ok is False

    def test_generate_error_includes_understandable_reason(self):
        """CA-7.2: El fallo debe incluir un motivo entendible del problema."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))
        resp = client.generate("test prompt")

        assert resp.error is not None
        assert len(resp.error) > 0

    def test_generate_does_not_raise_on_connection_error(self):
        """CA-7.3: El sistema no debe interrumpirse con un error técnico no controlado ante esta situación."""
        client = self.client_cls(self.config_cls(base_url="http://localhost:19999"))

        try:
            resp = client.generate("test prompt")
        except Exception as e:
            assert False, f"generate() no debería lanzar excepción, lanzó: {e}"

        assert resp.text == ""

    def test_llmconfig_centralizes_model_parameters(self):
        """CA-8.1: El sistema debe permitir centralizar en un solo lugar los parámetros de configuración del modelo."""
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
        """CA-8.2: Debe existir un conjunto de valores por defecto razonables si no se especifica configuración propia."""
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
        """CA-8.3: La configuración debe poder reutilizarse al crear distintas instancias del cliente"""
        config = self.config_cls(model="mistral:7b")

        client_a = self.client_cls(config)
        client_b = self.client_cls(config)

        assert client_a.config is config
        assert client_b.config is config
        assert client_a.config.model == client_b.config.model == "mistral:7b"

    def test_llmresponse_same_structure_success_and_failure(self):
        """CA-9.1: El sistema debe entregar siempre un resultado con la misma estructura,
        sin importar si la generación fue exitosa o falló."""
        from src.llm.client import LLMResponse

        success_fields = set(LLMResponse(text="ok", model="llama3:8b").__dict__.keys())
        failure_fields = set(
            LLMResponse(
                text="", model="llama3:8b", ok=False, error="algo falló"
            ).__dict__.keys()
        )

        assert success_fields == failure_fields

    def test_llmresponse_includes_generated_text_or_empty(self):
        """CA-9.2: El resultado debe incluir el texto generado (o vacío si falló)."""
        from src.llm.client import LLMResponse

        ok_resp = LLMResponse(text="Hola mundo", model="llama3:8b")
        fail_resp = LLMResponse(text="", model="llama3:8b", ok=False, error="error")

        assert ok_resp.text == "Hola mundo"
        assert fail_resp.text == ""

    @patch("requests.post")
    def test_llmresponse_includes_tokens_and_elapsed_time(self, mock_post):
        """CA-9.3: El resultado debe incluir las métricas de uso (tokens) y el tiempo que tomó la generación."""
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
        """CA-9.4: El resultado debe indicar claramente si fue exitoso o no, y por qué en caso de falla."""
        from src.llm.client import LLMResponse

        ok_resp = LLMResponse(text="Hola mundo", model="llama3:8b")
        fail_resp = LLMResponse(
            text="", model="llama3:8b", ok=False, error="Ollama no disponible"
        )

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

    @patch("src.llm.client.requests.post")
    def test_generate_sends_seed_when_configured(self, mock_post):
        from src.llm.client import LLMClient, LLMConfig

        """CA-11.1:  Si se indica ese número, tiene que llegarle efectivamente al motor de generación
        cuando se responde de una sola vez."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "ok", "eval_count": 1, "prompt_eval_count": 1},
        )
        mock_post.return_value.raise_for_status = lambda: None

        client = LLMClient(LLMConfig(seed=123))
        client.generate("pregunta")

        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["options"]["seed"] == 123

    @patch("src.llm.client.requests.post")
    def test_generate_omits_seed_when_not_configured(self, mock_post):
        from src.llm.client import LLMClient, LLMConfig

        """CA-12.1: Si no se indica ese número, no se le debe mandar nada raro al motor de generación 
        cuando se responde de una sola vez."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "ok", "eval_count": 1, "prompt_eval_count": 1},
        )
        mock_post.return_value.raise_for_status = lambda: None

        client = LLMClient(LLMConfig())
        client.generate("pregunta")

        sent_payload = mock_post.call_args.kwargs["json"]
        assert "seed" not in sent_payload["options"]
