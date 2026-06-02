import json
from dataclasses import asdict
from pathlib import Path

from .models import ProcessedDoc

_MANIFEST_EXCLUDE = frozenset({"text", "pages"})


def _doc_to_dict(doc: ProcessedDoc) -> dict:
    d = asdict(doc)
    d["pages"] = [{"page_num": p["page_num"], "text": p["text"]} for p in d["pages"]]
    return d


def save_doc(doc: ProcessedDoc, processed_dir: Path) -> Path:
    out = processed_dir / f"{doc.doc_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(_doc_to_dict(doc), f, ensure_ascii=False, indent=2)

    return out


def append_manifest(doc: ProcessedDoc, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {k: v for k, v in _doc_to_dict(doc).items() if k not in _MANIFEST_EXCLUDE}

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
