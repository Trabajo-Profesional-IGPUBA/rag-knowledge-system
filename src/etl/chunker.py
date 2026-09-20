"""
Módulo de chunking: divide el texto de un documento en fragmentos indexables.

Estrategia: HybridChunker de Docling, tiene en cuenta la jerarquia del documento al momento
de crear los chunks.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from src.etl.cleaner import normalize

logger = logging.getLogger()
"""
Importante: MAX_TOKENS debe coincidir con el max_seq_length real del
modelo de embedding usado (ver embedder.py), no con el límite teórico
de tokens del tokenizer subyacente. Un desalineamiento acá causa
truncado silencioso al embeddear (ver issue de migración a e5-base).
"""
MAX_TOKENS = 500

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

_PATTERNS: dict[str, re.Pattern] = {
    "well": re.compile(
        r"(?:pozo)\s*[:\-–]?\s*"
        r"([A-Z]{1,4}-\d{1,4}[A-Z]?"
        r"|\d{3,4}/\d{1,2}-\d{1,2})",
        re.IGNORECASE,
    ),
    "section": re.compile(r"(?:seccion|sección)[:\s]+(.+?)(?:\n|$)", re.IGNORECASE),
}


@dataclass
class ChunkMeta:
    well: str | None
    section: str | None


@dataclass
class Chunk:
    text: str
    contextualized_text: str
    source_file: str
    source_hash: str
    page: int | None
    section: str | None
    chunk_index: int
    well: str | None

    @property
    def meta(self):
        return {
            "well": self.well,
            "section": self.section,
            "source_file": self.source_file,
            "source_hash": self.source_hash,
            "page": self.page,
        }

    @property
    def chunk_id(self):
        return f"{self.source_hash}::{self.chunk_index}"

    @property
    def char_count(self):
        return len(self.text)


class DoclingHybridChunker:
    """
    Convierte un PDF en una secuencia de Chunks usando el HybridChunker de Docling.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        max_tokens: int = MAX_TOKENS,
    ):
        """
        Inicializa el chunker

        Es importante que el modelo de embedding utilizado en el tokenizer sea el mismo que el del embedder.
        """
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embedding_model),
            max_tokens=max_tokens,
        )
        self._chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        self._converter = DocumentConverter()
        logger.debug("Chunker initialized successfully")

    def chunk(self, filename: Path):
        """
        Parsea un PDF y genera Chunks con metadatos.
        """

        doc = self._converter.convert(filename).document
        for index, chunk in enumerate(self._chunker.chunk(dl_doc=doc)):
            contextualized_text = self._chunker.contextualize(chunk)
            # al contener datos de la jerarquia del documento conviene usar el texto contextualizado
            # para extraer la metadata
            meta = self.extract_chunk_metadata(contextualized_text)
            page = self.extract_page_no(chunk)

            yield Chunk(
                source_file=str(filename),
                source_hash=str(chunk.meta.origin.binary_hash),  # type: ignore
                text=normalize(chunk.text),
                contextualized_text=contextualized_text,
                chunk_index=index,
                page=page,
                section=meta.section,
                well=meta.well,
            )

    @staticmethod
    def extract_page_no(chunk):
        try:
            page = chunk.meta.doc_items[0].prov[0].page_no  # type: ignore
        except IndexError:
            page = None
        return page

    @staticmethod
    def extract_chunk_metadata(text: str) -> ChunkMeta:
        well_m = _PATTERNS["well"].search(text)
        well = well_m.group(1).strip() if well_m is not None else None

        section_m = _PATTERNS["section"].search(text)
        section = section_m.group(1).strip() if section_m is not None else None

        return ChunkMeta(
            well=well,
            section=section,
        )
