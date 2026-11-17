"""Turn a document's blocks into display pieces, with the cited span marked exactly.

The template escapes every piece of text; nothing here produces HTML.
"""

from rag.models import Block, Document


def _parts(block: Block, start: int | None, end: int | None) -> list[dict]:
    """A block's text split into plain and highlighted runs, using the citation's character offsets."""
    if start is None or end is None or block.end <= start or block.start >= end:
        return [{"text": block.text, "mark": False}]
    a = max(start, block.start) - block.start
    b = min(end, block.end) - block.start
    runs = [{"text": block.text[:a], "mark": False}, {"text": block.text[a:b], "mark": True},
            {"text": block.text[b:], "mark": False}]
    return [r for r in runs if r["text"]]


def _cells(row_text: str) -> list[tuple[str, str]]:
    """'City tier: Tier 1 | Maximum room rate per night: $275' -> [('City tier', 'Tier 1'), ...]."""
    cells = []
    for cell in row_text.split(" | "):
        key, sep, value = cell.partition(": ")
        cells.append((key, value) if sep else ("", cell))
    return cells


def render_blocks(doc: Document, start: int | None = None, end: int | None = None) -> list[dict]:
    """Display pieces: headings, paragraphs, lists (consecutive items) and tables (consecutive rows)."""
    pieces: list[dict] = []
    for block in doc.blocks:
        parts = _parts(block, start, end)
        marked = any(p["mark"] for p in parts)
        if block.kind == "heading":
            pieces.append({"type": "heading", "level": min(len(block.section) + 2, 4), "parts": parts})
        elif block.kind == "list_item":
            if not pieces or pieces[-1]["type"] != "list":
                pieces.append({"type": "list", "items": []})
            pieces[-1]["items"].append({"parts": parts, "marked": marked})
        elif block.kind == "table_row":
            cells = _cells(block.text)
            headers = [k for k, _ in cells]
            if pieces and pieces[-1]["type"] == "table" and pieces[-1]["headers"] == headers:
                pieces[-1]["rows"].append({"cells": [v for _, v in cells], "marked": marked})
            else:
                pieces.append({"type": "table", "headers": headers,
                               "show_headers": any(headers),
                               "rows": [{"cells": [v for _, v in cells], "marked": marked}]})
        else:
            pieces.append({"type": "paragraph", "parts": parts})
    return pieces
