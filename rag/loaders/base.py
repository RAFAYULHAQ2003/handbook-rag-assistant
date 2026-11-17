import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.models import Block


class LoaderError(Exception):
    """The file cannot be turned into a usable document."""


class UnsupportedFormat(LoaderError):
    pass


class EmptyDocument(LoaderError):
    """No text could be extracted (for example a scanned PDF)."""


@dataclass
class RawBlock:
    kind: str  # "heading" | "paragraph" | "list_item" | "table_row"
    text: str
    level: int = 0  # heading level, 1 = top


@dataclass
class ParsedFile:
    title: str | None = None
    meta: dict[str, str] = field(default_factory=dict)
    blocks: list[RawBlock] = field(default_factory=list)


# "Document ID: x | Owner: y | Effective: 2024-01-01 | Status: current | Supersedes: z"
HEADER_KEYS = {
    "document id": "doc_id",
    "owner": "owner",
    "effective": "effective_date",
    "status": "status",
    "supersedes": "supersedes",
    "category": "category",
}


def parse_header_line(line: str) -> dict[str, str]:
    """Parse the pipe-separated metadata line used by the PDF and Word documents."""
    meta = {}
    for part in line.split("|"):
        key, sep, value = part.partition(":")
        name = HEADER_KEYS.get(key.strip().lower())
        if sep and name and value.strip():
            meta[name] = value.strip()
    return meta


def is_header_line(line: str) -> bool:
    return line.strip().lower().startswith("document id:")


def clean_inline(text: str) -> str:
    """Collapse whitespace (including non-breaking spaces) to single spaces."""
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def read_text_file(path: Path) -> str:
    """Read a text file: UTF-8 (with or without BOM), falling back to Windows-1252.

    Binary data renamed to a text extension is rejected rather than decoded into garbage.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    high = sum(b >= 0x80 for b in data)
    if b"\x00" in data or high > 0.3 * len(data):
        raise LoaderError("file looks binary, not text")
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError as exc:
        raise LoaderError("file is neither UTF-8 nor Windows-1252 text") from exc


def table_row_text(headers: list[str], cells: list[str]) -> str:
    """'City tier: Tier 1 | Maximum room rate per night: $275' keeps each row self-explanatory.

    Cells beyond the header row are kept (without a label), never dropped.
    """
    headers = list(headers) + [""] * (len(cells) - len(headers))
    pairs = [f"{h}: {c}" if h else c for h, c in zip(headers, cells, strict=False) if c]
    return " | ".join(pairs)


def build_text(blocks: list[RawBlock], title: str | None = None) -> tuple[str, list[Block]]:
    """Join raw blocks into one text and compute offsets and heading paths.

    A leading heading that just repeats the title is dropped.
    """
    blocks = [b for b in blocks if b.text]
    if blocks and title and blocks[0].kind == "heading" and blocks[0].text.lower() == title.lower():
        blocks = blocks[1:]

    parts: list[str] = []
    out: list[Block] = []
    path: list[tuple[int, str]] = []  # (level, heading text)
    pos = 0
    for raw in blocks:
        if raw.kind == "heading":
            path = [p for p in path if p[0] < raw.level]
            section = tuple(t for _, t in path)
            path.append((raw.level, raw.text))
        else:
            section = tuple(t for _, t in path)
        if parts:
            parts.append("\n")
            pos += 1
        out.append(Block(raw.kind, raw.text, section, pos, pos + len(raw.text)))
        parts.append(raw.text)
        pos += len(raw.text)
    return "".join(parts), out
