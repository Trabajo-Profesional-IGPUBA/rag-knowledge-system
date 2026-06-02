import argparse
import logging
import sys
from pathlib import Path

from src.etl import run, BatchResult


ROOT_DIR      = Path(__file__).parent
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MANIFEST_PATH = ROOT_DIR / "data" / "manifest.jsonl"
LOG_DIR       = ROOT_DIR / "logs"


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    log_file = log_dir / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger(__name__).info(f"Log: {log_file}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline ETL: lee PDFs de data/raw/ y los procesa a data/processed/"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="PDFs por batch (default: 16)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Workers paralelos (default: nro de CPUs disponibles)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Lista los PDFs pendientes sin procesar nada"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Reprocesa todos los PDFs aunque ya existan los .json"
    )
    return parser.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    setup_logging(LOG_DIR)

    log = logging.getLogger(__name__)
    log.info("═" * 52)
    log.info("Iniciando pipeline ETL")
    log.info(f"  raw_dir      : {RAW_DIR}")
    log.info(f"  processed_dir: {PROCESSED_DIR}")
    log.info(f"  manifest     : {MANIFEST_PATH}")
    log.info(f"  batch_size   : {args.batch_size}")
    log.info(f"  workers      : {args.workers or 'auto'}")
    log.info(f"  incremental  : {not args.full}")
    log.info(f"  dry_run      : {args.dry_run}")
    log.info("═" * 52)

    if not RAW_DIR.exists():
        log.error(f"No existe data/raw/ en {RAW_DIR}. Creá la carpeta y colocá los PDFs.")
        return 1

    result: BatchResult = run(
        raw_dir       = RAW_DIR,
        processed_dir = PROCESSED_DIR,
        manifest_path = MANIFEST_PATH,
        batch_size    = args.batch_size,
        max_workers   = args.workers,
        incremental   = not args.full,
        dry_run       = args.dry_run,
    )

    if result.total_errors > 0:
        log.warning(f"{result.total_errors} PDFs fallaron. Revisá los logs.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
