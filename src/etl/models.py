from dataclasses import dataclass, field

@dataclass
class PageData:
    page_num: int
    text: str

@dataclass
class ProcessedDoc:
    doc_id:       str
    source_path:  str
    doc_type:     str
    filename:     str
    page_count:   int
    char_count:   int
    text:         str
    pages:        list[PageData]
    extracted_at: str
    error:        str | None = None

    def ok(self) -> bool:
        return self.error is None
