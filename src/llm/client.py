import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:8b"
DEFAULT_TIMEOUT = 120


@dataclass
class LLMResponse:
    """Resultado de una generación: texto, tokens, tiempo y estado de éxito/error."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_sec: float = 0.0
    ok: bool = True
    error: str | None = None


@dataclass
class LLMConfig:
    """Parámetros de configuración del modelo y de conexión al servidor Ollama."""

    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 1024
    repeat_penalty: float = 1.1
    seed: int | None = None
    base_url: str = OLLAMA_BASE_URL
    timeout: int = DEFAULT_TIMEOUT


class LLMClient:
    """
    Cliente para Ollama con soporte de streaming y logging.
    Diseñado para ser reemplazable por cualquier API compatible (OpenAI, etc.).
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Inicializa el cliente con la configuración dada (o los defaults si no se especifica)."""
        self.config = config or LLMConfig()
        self._base_url = self.config.base_url.rstrip("/")
        log.info(
            "LLMClient inicializado — modelo: %s, url: %s",
            self.config.model,
            self._base_url,
        )

    def is_available(self) -> bool:
        """Verifica que Ollama esté corriendo."""
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Lista modelos disponibles en Ollama."""
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            log.warning("No se pudieron listar modelos: %s", e)
            return []

    def pull_model(self, model: str | None = None) -> bool:
        """Descarga un modelo si no está disponible."""
        model = model or self.config.model
        log.info("Descargando modelo %s (puede tardar varios minutos)...", model)
        try:
            r = requests.post(
                f"{self._base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=600,
            )
            r.raise_for_status()
            log.info("Modelo %s descargado correctamente", model)
            return True
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            log.error("Error descargando modelo %s: %s", model, e)
            return False

    def generate(self, prompt: str, model: str | None = None) -> LLMResponse:
        """
        Genera una respuesta completa (no streaming).

        Args:
            prompt: texto del prompt completo (sistema + contexto + pregunta).
            model: nombre del modelo (usa config.model si no se especifica).

        Returns:
            LLMResponse con el texto generado y métricas.
        """
        model = model or self.config.model
        t0 = time.perf_counter()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(),
        }

        try:
            r = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout,
            )
            r.raise_for_status()
            data = r.json()

            elapsed = time.perf_counter() - t0
            text = data.get("response", "").strip()

            log.info(
                "LLM generó respuesta — modelo=%s, tokens=%d, tiempo=%.2fs",
                model,
                data.get("eval_count", 0),
                elapsed,
            )

            return LLMResponse(
                text=text,
                model=model,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                elapsed_sec=elapsed,
            )

        except requests.exceptions.ConnectionError:
            msg = f"Ollama no disponible en {self._base_url}. ¿Está corriendo? Ejecutá: ollama serve"
            log.error(msg)
            return LLMResponse(text="", model=model, ok=False, error=msg)

        except (
            requests.exceptions.RequestException,
            json.JSONDecodeError,
            KeyError,
        ) as e:
            msg = str(e)
            log.error("Error en LLM generate: %s", msg)
            return LLMResponse(text="", model=model, ok=False, error=msg)

    def generate_stream(self, prompt: str, model: str | None = None) -> Iterator[str]:
        """Genera una respuesta en modo streaming, yieldeando tokens a medida que llegan."""
        model = model or self.config.model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.num_predict,
                "repeat_penalty": self.config.repeat_penalty,
            },
        }

        try:
            with requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.config.timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            log.warning(
                                "Línea de streaming inválida, se ignora: %s", line
                            )
                            continue

                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break

        except requests.exceptions.ConnectionError:
            yield "\n[Error: Ollama no disponible. Ejecutá: ollama serve]\n"
        except requests.exceptions.RequestException as e:
            yield f"\n[Error: {e}]\n"

    def _build_options(self) -> dict:
        """Arma el dict de 'options' para el payload de Ollama, incluyendo
        seed solo si fue especificado (CA-2: no mandar 'seed': None)."""
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_predict": self.config.num_predict,
            "repeat_penalty": self.config.repeat_penalty,
        }
        if self.config.seed is not None:
            options["seed"] = self.config.seed
        return options
