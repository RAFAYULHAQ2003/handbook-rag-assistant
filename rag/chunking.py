import re
from dataclasses import dataclass

from rag.models import Block, Document

MAX_CHARS = 1000  # prose passage limit, roughly 200-250 tokens
TABLE_MAX_CHARS = 2000  # whole-table passage limit; bigger tables become groups of rows

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Passage:
    passage_id: str  # "<doc_id>#<n>", stable for the same document text
    doc_id: str
    kind: str  # "text" | "row" | "table"
    section: tuple[str, ...]
    text: str  # exactly document.text[start:end]
    start: int
    end: int
    search_text: str  # title and section path + text, used for indexing


def _context(doc: Document, section: tuple[str, ...]) -> str:
    return " > ".join((doc.meta.title, *section))


def _spans_by_size(blocks: list[Block], limit: int) -> list[list[Block]]:
    """Pack consecutive blocks into groups no longer than `limit` characters."""
    groups: list[list[Block]] = []
    current: list[Block] = []
    for b in blocks:
        if current and (b.end - current[0].start) > limit:
            groups.append(current)
            current = []
        current.append(b)
    if current:
        groups.append(current)
    return groups


def _sentence_spans(block: Block, limit: int) -> list[tuple[int, int]]:
    """Split one over-long block into (start, end) spans at sentence boundaries."""
    cuts = [block.start] + [block.start + m.end() for m in _SENTENCE_BREAK.finditer(block.text)] + [block.end]
    spans: list[tuple[int, int]] = []
    span_start = cuts[0]
    for prev, cut in zip(cuts, cuts[1:], strict=False):
        if cut - span_start > limit and prev > span_start:
            spans.append((span_start, prev))
            span_start = prev
    spans.append((span_start, block.end))
    return spans  # whitespace left at span edges is trimmed when the passage is added


def _runs(doc: Document) -> list[tuple[str, list[Block]]]:
    """Consecutive non-heading blocks of the same kind group ("prose" or "table").

    A heading always ends a run, and a section only changes at a heading, so
    every run lies within one section.
    """
    runs: list[tuple[str, list[Block]]] = []
    for b in doc.blocks:
        if b.kind == "heading":
            runs.append(("break", []))
            continue
        group = "table" if b.kind == "table_row" else "prose"
        if runs and runs[-1][0] == group and runs[-1][1]:
            runs[-1][1].append(b)
        else:
            runs.append((group, [b]))
    return [(g, blocks) for g, blocks in runs if blocks]


def chunk_document(doc: Document) -> list[Passage]:
    passages: list[Passage] = []

    def add(kind: str, section: tuple[str, ...], start: int, end: int):
        # strip whitespace at the edges so text == document.text[start:end] stays clean
        raw = doc.text[start:end]
        start += len(raw) - len(raw.lstrip())
        end -= len(raw) - len(raw.rstrip())
        if end <= start:
            return
        text = doc.text[start:end]
        passages.append(Passage(
            passage_id=f"{doc.doc_id}#{len(passages)}",
            doc_id=doc.doc_id,
            kind=kind,
            section=section,
            text=text,
            start=start,
            end=end,
            search_text=f"{_context(doc, section)}\n{text}",
        ))

    runs = _runs(doc)
    for i, (group, blocks) in enumerate(runs):
        section = blocks[0].section
        if group == "prose":
            for span in _spans_by_size(blocks, MAX_CHARS):
                if len(span) == 1 and len(span[0].text) > MAX_CHARS:
                    for s, e in _sentence_spans(span[0], MAX_CHARS):
                        add("text", section, s, e)
                else:
                    add("text", section, span[0].start, span[-1].end)
            continue

        for row in blocks:
            add("row", section, row.start, row.end)
        # whole-table passages, led by the sentence that introduces the table when there is one
        prev = runs[i - 1] if i > 0 else None
        lead = prev[1][-1] if prev and prev[0] == "prose" and prev[1][-1].section == section else None
        for g, rows in enumerate(_spans_by_size(blocks, TABLE_MAX_CHARS)):
            start = lead.start if lead is not None and g == 0 else rows[0].start
            add("table", section, start, rows[-1].end)
    return passages


def chunk_documents(docs: list[Document]) -> list[Passage]:
    return [p for doc in docs for p in chunk_document(doc)]
