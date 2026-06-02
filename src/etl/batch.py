import logging
import time
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .reader import extract
from .writer import save_doc, append_manifest_batch, _to_dict, _EXCLUDE

log = logging.getLogger(__name__)


@dataclass
class BatchResult:
    total_found: int = 0
    total_skipped: int = 0
    total_ok: int = 0
    total_errors: int = 0
    elapsed_sec: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)


def discover_stream(raw_dir: Path) -> Iterator[Path]:
    yield from raw_dir.rglob("*.pdf")


def is_processed(pdf: Path, raw_dir: Path, processed_dir: Path) -> bool:
    rel = pdf.relative_to(raw_dir).with_suffix("")
    return (processed_dir / f"{rel}.json").exists()


def _worker(pdf: Path, raw_dir: Path):
    doc = extract(pdf, raw_dir)
    return pdf, doc


def run(
    raw_dir: Path,
    processed_dir: Path,
    manifest_path: Path,
    max_workers: int | None = None,
    incremental: bool = True,
    manifest_buffer_size: int = 500,
    in_flight: int | None = None,
) -> BatchResult:

    result = BatchResult()
    t0 = time.monotonic()

    manifest_buffer: list[dict] = []

    max_workers = max_workers or os.cpu_count()
    in_flight = in_flight or max_workers * 2

    active = set()
    pdf_iter = discover_stream(raw_dir)

    def submit_next():
        for pdf in pdf_iter:
            result.total_found += 1

            if incremental and is_processed(pdf, raw_dir, processed_dir):
                result.total_skipped += 1
                continue

            fut = executor.submit(_worker, pdf, raw_dir)
            active.add(fut)
            return True

        return False

    def flush_manifest():
        nonlocal manifest_buffer
        if manifest_buffer:
            append_manifest_batch(manifest_buffer, manifest_path)
            manifest_buffer.clear()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for _ in range(in_flight):
            if not submit_next():
                break

        while active:

            done, active = wait(active, return_when=FIRST_COMPLETED)

            for fut in done:
                pdf, doc = fut.result()

                if not doc.ok():
                    result.total_errors += 1
                    result.errors.append((str(pdf), str(doc.error or "")))
                else:
                    save_doc(doc, processed_dir)

                    meta = {k: v for k, v in _to_dict(doc).items() if k not in _EXCLUDE}

                    manifest_buffer.append(meta)

                    if len(manifest_buffer) >= manifest_buffer_size:
                        flush_manifest()

                    result.total_ok += 1

                submit_next()

        flush_manifest()

    result.elapsed_sec = time.monotonic() - t0
    return result
