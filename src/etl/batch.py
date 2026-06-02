import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .models import ProcessedDoc
from .reader import extract
from .writer import append_manifest, save_doc

log = logging.getLogger(__name__)


# Resultado del run


@dataclass
class BatchResult:
    total_found: int = 0
    total_skipped: int = 0
    total_ok: int = 0
    total_errors: int = 0
    elapsed_sec: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.total_ok + self.total_errors


def discover(
    raw_dir: Path,
    processed_dir: Path,
    incremental: bool = True,
) -> tuple[list[Path], int]:
    all_pdfs = sorted(raw_dir.rglob("*.pdf"))
    if not incremental:
        return all_pdfs, 0

    pending = []
    skipped = 0
    for p in all_pdfs:
        rel = p.relative_to(raw_dir).with_suffix("")
        output = processed_dir / f"{rel}.json"
        if output.exists():
            skipped += 1
        else:
            pending.append(p)

    return pending, skipped


# Helper para workers


def _worker(args: tuple[Path, Path]) -> ProcessedDoc:
    pdf_path, raw_dir = args
    return extract(pdf_path, raw_dir)


# Orquestador principal


def run(
    raw_dir: Path,
    processed_dir: Path,
    manifest_path: Path,
    batch_size: int = 16,
    max_workers: int | None = None,
    incremental: bool = True,
    dry_run: bool = False,
) -> BatchResult:

    result = BatchResult()
    t0 = time.monotonic()

    pending, skipped = discover(raw_dir, processed_dir, incremental)

    result.total_found = len(pending) + skipped
    result.total_skipped = skipped

    log.info(f"PDFs encontrados : {result.total_found}")
    log.info(f"Ya procesados    : {skipped}")
    log.info(f"Pendientes       : {len(pending)}")
    log.info(f"Batch size       : {batch_size} | Workers: {max_workers or 'auto'}")

    if dry_run:
        log.info("── DRY RUN ── solo listado, no se procesa nada")
        for p in pending:
            log.info(f"  pendiente: {p.relative_to(raw_dir)}")
        result.elapsed_sec = time.monotonic() - t0
        return result

    if not pending:
        log.info("Nada que procesar.")
        result.elapsed_sec = time.monotonic() - t0
        return result

    batches = _chunks(pending, batch_size)
    total_batches = len(batches)

    for batch_num, batch in enumerate(batches, start=1):
        log.info(f"── Batch {batch_num}/{total_batches} ({len(batch)} PDFs) ──")
        t_batch = time.monotonic()

        args = [(pdf, raw_dir) for pdf in batch]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, a): a[0] for a in args}

            for future in as_completed(futures):
                pdf_path = futures[future]

                try:
                    doc = future.result()
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    log.error(f"  CRASH  {pdf_path.name}: {msg}")
                    result.total_errors += 1
                    result.errors.append((str(pdf_path), msg))
                    continue

                if not doc.ok():
                    log.error(f"  ERROR  {doc.source_path}: {doc.error}")
                    result.total_errors += 1
                    result.errors.append((doc.source_path, doc.error or ""))
                else:
                    save_doc(doc, processed_dir)
                    append_manifest(doc, manifest_path)
                    log.info(
                        f"  OK  {doc.source_path}"
                        f" | {doc.page_count}p | {doc.char_count:,} chars"
                    )
                    result.total_ok += 1

        log.info(f"  Batch {batch_num} en {time.monotonic() - t_batch:.1f}s")

    result.elapsed_sec = time.monotonic() - t0

    log.info("═" * 52)
    log.info(
        f"OK: {result.total_ok}  |  Errores: {result.total_errors}"
        f"  |  Tiempo: {result.elapsed_sec:.1f}s"
    )
    log.info("═" * 52)

    return result


# Utilidades


def _chunks(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]
