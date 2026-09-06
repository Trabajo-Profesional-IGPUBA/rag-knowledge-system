"""
Módulo de observabilidad y monitoreo.

Provee:
  - Setup de logging estructurado con rotación de archivos.
  - Métricas básicas del pipeline (tiempo, volumen, errores).
  - Reporte de estado del sistema.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from pathlib import Path
from typing import Self

EXTERNAL_LOGGERS = ["docling", "httpx", "rapidocr"]


def setup_logging(
    log_dir: Path,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configura logging con rotación de archivos y salida a consola.

    Args:
        log_dir: directorio donde se guardan los logs.
        level: nivel de logging (default INFO).
        max_bytes: tamaño máximo de cada archivo de log.
        backup_count: cantidad de archivos de backup a mantener.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "rag_pipeline.log"

    for logger_name in EXTERNAL_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ]

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


class Timer:
    """Context manager para medir tiempos."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start
