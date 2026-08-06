"""
Módulo de observabilidad y monitoreo.

Provee:
  - Setup de logging estructurado con rotación de archivos.
  - Métricas básicas del pipeline (tiempo, volumen, errores).
  - Reporte de estado del sistema.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


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


@dataclass
class PipelineMetrics:
    """Acumula métricas del pipeline completo."""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""

    # ETL
    docs_found: int = 0
    docs_ok: int = 0
    docs_error: int = 0
    docs_skipped: int = 0
    ocr_pages: int = 0
    etl_elapsed_sec: float = 0.0

    # Chunking
    chunks_generated: int = 0

    # Embeddings
    embeddings_generated: int = 0
    embeddings_elapsed_sec: float = 0.0

    # Vectorstore
    vectors_indexed: int = 0

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        self.finish()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logging.getLogger(__name__).info("Métricas guardadas en %s", path)

    def log_summary(self) -> None:
        log = logging.getLogger(__name__)
        log.info("═" * 52)
        log.info("RESUMEN DEL PIPELINE")
        log.info("  ETL     → OK=%d ERR=%d SKIP=%d OCR_PAGES=%d (%.2fs)",
                 self.docs_ok, self.docs_error, self.docs_skipped,
                 self.ocr_pages, self.etl_elapsed_sec)
        log.info("  CHUNKS  → %d generados", self.chunks_generated)
        log.info("  EMBED   → %d vectores (%.2fs)", self.embeddings_generated, self.embeddings_elapsed_sec)
        log.info("  VSTORE  → %d indexados", self.vectors_indexed)
        log.info("═" * 52)


class Timer:
    """Context manager para medir tiempos."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._start
