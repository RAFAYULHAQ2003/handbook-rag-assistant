from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Block:
    """One structural unit of a document: a heading, paragraph, list item or table row.

    `start`/`end` are character offsets into `Document.text`, so a cited block
    can be highlighted exactly: `document.text[block.start:block.end] == block.text`.
    """

    kind: str  # "heading" | "paragraph" | "list_item" | "table_row"
    text: str
    section: tuple[str, ...]  # heading path the block sits under, e.g. ("4. Limits", "4.2 Hotels")
    start: int
    end: int


@dataclass
class DocumentMeta:
    doc_id: str
    title: str
    path: str  # file name relative to the corpus directory
    format: str  # "md" | "pdf" | "docx" | "csv" | "txt" | "html"
    category: str | None = None
    owner: str | None = None
    effective_date: date | None = None
    status: str | None = None  # as declared by the manifest or file; see rag.validity for the real state
    supersedes: str | None = None
    in_manifest: bool = False
    uploaded: bool = False  # added through the web page: never official, never replaces a handbook document


@dataclass
class Document:
    meta: DocumentMeta
    text: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.meta.doc_id


@dataclass(frozen=True)
class MetadataFinding:
    """A disagreement or gap between the manifest and a file's own metadata."""

    doc_id: str
    field: str
    manifest_value: str | None
    file_value: str | None
