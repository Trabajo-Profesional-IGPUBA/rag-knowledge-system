import argparse
import logging
import sys
from pathlib import Path

from src.etl import BatchResult, run

ROOT_DIR = Path(__file__).parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MANIFEST_PATH = ROOT_DIR / "data" / "manifest.jsonl"
LOG_DIR = ROOT_DIR / "logs"


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    log_file = (
        log_dir / f"etl_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL PDF pipeline")

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers (default CPU cores)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Reprocesar todo ignorando incremental",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(LOG_DIR)

    log = logging.getLogger(__name__)

    log.info("═" * 52)
    log.info("Iniciando pipeline ETL")
    log.info(f"raw_dir      : {RAW_DIR}")
    log.info(f"processed_dir: {PROCESSED_DIR}")
    log.info(f"manifest     : {MANIFEST_PATH}")
    log.info(f"workers      : {args.workers or 'auto'}")
    log.info(f"incremental  : {not args.full}")
    log.info("═" * 52)

    if not RAW_DIR.exists():
        log.error(f"No existe {RAW_DIR}")
        return 1

    result: BatchResult = run(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        manifest_path=MANIFEST_PATH,
        max_workers=args.workers,
        incremental=not args.full,
    )

    log.info(
        f"FIN → OK={result.total_ok} "
        f"ERR={result.total_errors} "
        f"SKIP={result.total_skipped} "
        f"TIME={result.elapsed_sec:.2f}s"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
