from __future__ import annotations

import json
from pathlib import Path


def cargar_documentos(data_dir: str = "data/processed") -> list[dict]:
    docs = []
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"No existe el directorio {data_dir}")

    for path in sorted(base.rglob("*.json")):
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  {path} no es JSON válido, se omite: {e}")
                continue
            data["_source_file"] = str(path)
            docs.append(data)

    return docs


def extraer_test_texts(docs: list[dict], max_docs: int | None = None) -> list[str]:
    textos = [d["text"] for d in docs if d.get("text")]
    if max_docs:
        textos = textos[:max_docs]
    return textos


def resumen_por_doc_type(docs: list[dict]) -> dict[str, int]:
    conteo: dict[str, int] = {}
    for d in docs:
        dt = d.get("doc_type", "desconocido")
        conteo[dt] = conteo.get(dt, 0) + 1
    return conteo


def detectar_posibles_duplicados(docs: list[dict]) -> list[tuple[str, str]]:
    por_texto: dict[str, list[str]] = {}
    for d in docs:
        texto = d.get("text", "").strip()
        if not texto:
            continue
        por_texto.setdefault(texto, []).append(d.get("doc_id", d.get("filename", "?")))

    duplicados = []
    for texto, ids in por_texto.items():
        if len(ids) > 1:
            for i in range(len(ids) - 1):
                duplicados.append((ids[i], ids[i + 1]))
    return duplicados
