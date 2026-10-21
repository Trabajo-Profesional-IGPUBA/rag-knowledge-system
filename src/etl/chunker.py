"""
Módulo de chunking: divide el texto de un documento en fragmentos indexables.

Estrategia: HybridChunker de Docling, tiene en cuenta la jerarquia del documento al momento
de crear los chunks.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from src.etl.chunk_meta import extract_wells
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


@dataclass
class Chunk:
    text: str
    contextualized_text: str
    source_file: str
    source_hash: str
    page: int | None
    chunk_index: int
    wells: list[str] | None

    @property
    def meta(self):
        return {
            "wells": self.wells,
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
        no_well_extraction: list[Path] = [],
        do_ocr: bool = False,
    ):
        """
        Inicializa el chunker

        Es importante que el modelo de embedding utilizado en el tokenizer sea el mismo que el del embedder.
        """
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embedding_model),
            max_tokens=max_tokens,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr

        self._chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
        self._no_well_extraction = [Path(p).resolve() for p in no_well_extraction]
        logger.debug("Chunker initialized successfully")

    def _should_extract_well(self, filename: Path) -> bool:
        resolved = Path(filename).resolve()
        return not any(
            resolved == excluded or excluded in resolved.parents
            for excluded in self._no_well_extraction
        )

    def chunk(self, filename: Path, source_file: Path | None = None):
        """
        Parsea un PDF y genera Chunks con metadatos.

        Para archivos grandes que son partidos previamente, se usa `source_file` en la metadata.
        """

        doc = self._converter.convert(filename).document
        should_extract_well_ids = self._should_extract_well(filename)
        for index, chunk in enumerate(self._chunker.chunk(dl_doc=doc)):
            contextualized_text = self._chunker.contextualize(chunk)
            # al contener datos de la jerarquia del documento conviene usar el texto contextualizado
            # para extraer la metadata
            wells = extract_wells(contextualized_text) or None
            page = self.extract_page_no(chunk)

            yield Chunk(
                source_file=str((source_file or filename).name),
                source_hash=str(chunk.meta.origin.binary_hash),  # type: ignore
                text=normalize(chunk.text),
                contextualized_text=contextualized_text,
                chunk_index=index,
                page=page,
                wells=wells if should_extract_well_ids else None,
            )

    @staticmethod
    def extract_page_no(chunk):
        try:
            page = chunk.meta.doc_items[0].prov[0].page_no  # type: ignore
        except IndexError:
            page = None
        return page
