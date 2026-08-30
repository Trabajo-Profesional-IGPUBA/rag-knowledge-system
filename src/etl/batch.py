import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from queue import Empty, Queue

from .reader import extract
from .writer import _EXCLUDE, _to_dict, append_manifest_batch, save_doc

log = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class BatchResult:
    total_found: int = 0
    total_skipped: int = 0
    total_ok: int = 0
    total_errors: int = 0
    elapsed_sec: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _open_index(processed_dir: Path) -> sqlite3.Connection:
    processed_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(processed_dir / "index.db")
    conn.execute("CREATE TABLE IF NOT EXISTS done (rel_path TEXT PRIMARY KEY)")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()
    return conn


def _sync_index_from_disk(processed_dir: Path, conn: sqlite3.Connection) -> None:
    rels = [
        str(p.relative_to(processed_dir).with_suffix(".pdf"))
        for p in processed_dir.rglob("*.json")
    ]
    if rels:
        conn.executemany(
            "INSERT OR IGNORE INTO done (rel_path) VALUES (?)",
            [(r,) for r in rels],
        )
        conn.commit()


def is_processed(pdf: Path, raw_dir: Path, conn: sqlite3.Connection) -> bool:
    rel = str(pdf.relative_to(raw_dir))
    return (
        conn.execute("SELECT 1 FROM done WHERE rel_path = ?", (rel,)).fetchone()
        is not None
    )


def _mark_processed(rels: list[str], conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO done (rel_path) VALUES (?)",
        [(r,) for r in rels],
    )
    conn.commit()


def _discover_worker(
    raw_dir: Path,
    processed_dir: Path,
    incremental: bool,
    queue: Queue,
    found_counter: list,
    skipped_counter: list,
) -> None:
    conn = _open_index(processed_dir)
    found = 0
    skipped = 0
    try:
        for pdf in raw_dir.rglob("*.pdf"):
            found += 1
            if incremental and is_processed(pdf, raw_dir, conn):
                skipped += 1
                continue
            queue.put(pdf)
    finally:
        found_counter[0] = found
        skipped_counter[0] = skipped
        conn.close()
        queue.put(_SENTINEL)


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
    discover_queue_size: int = 1000,
    max_logged_errors: int = 10_000,
) -> BatchResult:
    result = BatchResult()
    t0 = time.monotonic()

    manifest_buffer: list[dict] = []
    rels_to_mark: list[str] = []

    max_workers = max_workers or os.cpu_count()
    in_flight = in_flight or max_workers * 2

    conn = _open_index(processed_dir)
    _sync_index_from_disk(processed_dir, conn)

    found_counter = [0]
    skipped_counter = [0]

    pdf_queue: Queue = Queue(maxsize=discover_queue_size)

    discover_thread = threading.Thread(
        target=_discover_worker,
        args=(
            raw_dir,
            processed_dir,
            incremental,
            pdf_queue,
            found_counter,
            skipped_counter,
        ),
        daemon=True,
    )
    discover_thread.start()

    active: set = set()
    discover_exhausted = False

    def submit_next() -> bool:
        nonlocal discover_exhausted
        if discover_exhausted:
            return False
        while True:
            try:
                pdf = pdf_queue.get(timeout=1)
                break
            except Empty:
                if not discover_thread.is_alive():
                    log.warning("discover thread died without sending sentinel")
                    discover_exhausted = True
                    return False
        if pdf is _SENTINEL:
            discover_exhausted = True
            return False
        fut = executor.submit(_worker, pdf, raw_dir)
        active.add(fut)
        return True

    def flush_manifest() -> None:
        nonlocal manifest_buffer
        if manifest_buffer:
            append_manifest_batch(manifest_buffer, manifest_path)
            manifest_buffer.clear()

    def flush_index() -> None:
        nonlocal rels_to_mark
        if rels_to_mark:
            _mark_processed(rels_to_mark, conn)
            rels_to_mark.clear()

    def flush_all() -> None:
        flush_manifest()
        flush_index()

    def log_error(pdf_path: str, msg: str) -> None:
        result.total_errors += 1
        if len(result.errors) < max_logged_errors:
            result.errors.append((pdf_path, msg))
        else:
            log.error("error limit reached, logging only: %s — %s", pdf_path, msg)

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=get_context("forkserver")
    ) as executor:
        while len(active) < in_flight:
            if not submit_next():
                break

        while active or not discover_exhausted:
            if not active:
                if not submit_next():
                    break
                continue

            done, active = wait(active, return_when=FIRST_COMPLETED)

            for fut in done:
                try:
                    pdf, doc = fut.result()
                except Exception as e:
                    log_error("unknown", str(e))
                    continue

                if not doc.ok():
                    log_error(str(pdf), str(doc.error or ""))
                else:
                    save_doc(doc, processed_dir)
                    meta = {k: v for k, v in _to_dict(doc).items() if k not in _EXCLUDE}
                    manifest_buffer.append(meta)
                    rels_to_mark.append(str(pdf.relative_to(raw_dir)))

                    if len(manifest_buffer) >= manifest_buffer_size:
                        flush_all()

                    result.total_ok += 1

            while len(active) < in_flight:
                if not submit_next():
                    break

        flush_all()

    discover_thread.join()
    result.total_found = found_counter[0]
    result.total_skipped = skipped_counter[0]

    conn.close()
    result.elapsed_sec = time.monotonic() - t0
    return result
