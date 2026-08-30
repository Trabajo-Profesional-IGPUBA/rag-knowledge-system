import json
from pathlib import Path


def load_documents(data_dir: str = "data/processed") -> list[dict]:
    """Carga todos los JSON de data_dir (recursivamente) como lista de dicts, agregando la ruta de origen en '_source_file'."""
    docs = []
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist")

    for path in sorted(base.rglob("*.json")):
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  {path} is not valid JSON, skipping: {e}")
                continue
            data["_source_file"] = str(path)
            docs.append(data)

    return docs


def extract_test_texts(docs: list[dict], max_docs: int | None = None) -> list[str]:
    """Extrae el campo 'text' de cada documento (descartando vacíos), opcionalmente limitado a max_docs."""
    texts = [d["text"] for d in docs if d.get("text")]
    if max_docs:
        texts = texts[:max_docs]
    return texts


def summary_by_doc_type(docs: list[dict]) -> dict[str, int]:
    """Cuenta cuántos documentos hay por cada valor de 'doc_type' (usando 'unknown' si falta)."""
    count: dict[str, int] = {}
    for d in docs:
        dt = d.get("doc_type", "unknown")
        count[dt] = count.get(dt, 0) + 1
    return count


def detect_possible_duplicates(docs: list[dict]) -> list[tuple[str, str]]:
    """Detecta pares de documentos con texto idéntico y devuelve sus (doc_id, doc_id) como posibles duplicados."""
    by_text: dict[str, list[str]] = {}
    for d in docs:
        text = d.get("text", "").strip()
        if not text:
            continue
        by_text.setdefault(text, []).append(d.get("doc_id", d.get("filename", "?")))

    duplicates = []
    for text, ids in by_text.items():
        if len(ids) > 1:
            for i in range(len(ids) - 1):
                duplicates.append((ids[i], ids[i + 1]))
    return duplicates