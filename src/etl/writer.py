import json
from dataclasses import asdict
from pathlib import Path

from .models import ProcessedDoc

_EXCLUDE = {"text", "pages"}


def _to_dict(doc: ProcessedDoc) -> dict:
    d = asdict(doc)
    d["pages"] = [{"page_num": p["page_num"], "text": p["text"]} for p in d["pages"]]
    return d


def save_doc(doc: ProcessedDoc, processed_dir: Path) -> Path:
    out = processed_dir / f"{doc.doc_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(_to_dict(doc), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def append_manifest_batch(
    docs: ProcessedDoc | list[ProcessedDoc] | list[dict], manifest_path: Path
) -> None:

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(docs, list):
        docs = [docs]

    with open(manifest_path, "a", encoding="utf-8") as f:
        for doc in docs:

            if hasattr(doc, "__dataclass_fields__"):
                meta = {k: v for k, v in _to_dict(doc).items() if k not in _EXCLUDE}
            else:
                meta = doc

            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
