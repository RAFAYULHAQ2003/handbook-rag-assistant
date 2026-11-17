"""Turn any supported file into a ParsedFile; pick the loader by extension."""

from pathlib import Path

from rag.loaders import csv, docx, html, markdown, pdf, text
from rag.loaders.base import (
    EmptyDocument,
    LoaderError,
    ParsedFile,
    RawBlock,
    UnsupportedFormat,
    build_text,
)

LOADERS = {
    ".md": markdown.load,
    ".markdown": markdown.load,
    ".pdf": pdf.load,
    ".docx": docx.load,
    ".csv": csv.load,
    ".html": html.load,
    ".htm": html.load,
    ".txt": text.load,
}

SUPPORTED_EXTENSIONS = sorted(LOADERS)


def parse_file(path: Path) -> ParsedFile:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFormat(f"unsupported file type: {path.suffix or '(none)'}")
    parsed = loader(path)
    if not any(b.text.strip() for b in parsed.blocks):
        raise EmptyDocument(f"no text could be extracted from {path.name}")
    return parsed


__all__ = [
    "EmptyDocument",
    "LoaderError",
    "ParsedFile",
    "RawBlock",
    "SUPPORTED_EXTENSIONS",
    "UnsupportedFormat",
    "build_text",
    "parse_file",
]
