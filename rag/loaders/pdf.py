import re
from pathlib import Path

from pypdf import PdfReader

from rag.loaders.base import (
    LoaderError,
    ParsedFile,
    RawBlock,
    clean_inline,
    is_header_line,
    parse_header_line,
)

_BULLET_ONLY = {"\x7f", "•", "●", "▪", "◦", "‣", "-", "–", "*"}
_BULLET_PREFIX = re.compile(r"^(?:[\x7f•●▪◦‣]\s*|[-–*]\s+)(.+)$")
_NUMBER_ONLY = re.compile(r"^\d+[.)]?$")
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)
# "2. Annual leave", "4.2 Hotels": numbered, short, no full stop
_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.]{0,60})$")
_SENTENCE_END = (".", ":", "!", "?")


def extract_pages(path: Path) -> list[list[str]]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise LoaderError("PDF is password-protected")
        pages = [page.extract_text() or "" for page in reader.pages]
    except LoaderError:
        raise
    except Exception as exc:  # pypdf raises many error types on malformed files
        raise LoaderError(f"cannot read PDF: {exc}") from exc
    return [[line.strip() for line in page.splitlines()] for page in pages]


def drop_page_numbers(pages: list[list[str]]) -> list[str]:
    """Join pages, dropping a page number that ends (or starts) a page of a multi-page PDF."""
    lines: list[str] = []
    for page in pages:
        page = [line for line in page if line]
        if len(pages) > 1:
            if page and _PAGE_NUMBER.match(page[-1]):
                page = page[:-1]
            if page and _PAGE_NUMBER.match(page[0]) and page[0].lower().startswith("page"):
                page = page[1:]
        lines.extend(page)
    return lines


def split_header(lines: list[str]) -> tuple[str | None, dict[str, str], list[str]]:
    """Take the title and the metadata header (possibly wrapped) off the top.

    The first line is the title only when the metadata header follows it, or it
    reads like a title (short, no closing full stop); otherwise it is body text.
    """
    lines = [line for line in lines if line]
    title, meta = None, {}
    if lines and not is_header_line(lines[0]):
        next_is_header = len(lines) > 1 and is_header_line(lines[1])
        looks_like_title = len(lines[0]) <= 80 and not lines[0].endswith(_SENTENCE_END)
        if next_is_header or looks_like_title:
            title, lines = lines[0], lines[1:]
    if lines and is_header_line(lines[0]):
        header = lines.pop(0)
        # a header that ends on a dangling key ("Supersedes:") continues on the next line
        while lines and header.rstrip().endswith((":", "|")):
            header += " " + lines.pop(0)
        meta = parse_header_line(header)
    return title, meta, lines


def _join(lines: list[str]) -> str:
    """Join wrapped lines; re-attach words hyphenated across a line break."""
    text = ""
    for line in lines:
        if text.endswith("-") and line[:1].islower() and len(text) > 1 and text[-2].isalpha():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line
    return clean_inline(text)


def lines_to_blocks(lines: list[str]) -> list[RawBlock]:
    blocks: list[RawBlock] = []
    width = max((len(line) for line in lines), default=0)
    current: list[str] = []
    kind = "paragraph"

    def flush():
        nonlocal kind
        if current:
            blocks.append(RawBlock(kind, _join(current)))
            current.clear()
        kind = "paragraph"

    for line in lines:
        if line in _BULLET_ONLY or _NUMBER_ONLY.match(line):
            flush()  # marker on its own line: the following lines are one list item
            kind = "list_item"
            continue
        if m := _BULLET_PREFIX.match(line):
            flush()
            kind = "list_item"
            current.append(m.group(1).strip())
            continue
        if m := _HEADING.match(line):
            flush()
            blocks.append(RawBlock("heading", clean_inline(line), level=m.group(1).count(".") + 1))
            continue
        if current and current[-1].endswith(_SENTENCE_END) and len(current[-1]) < 0.8 * width:
            flush()  # previous line ended a sentence well short of the margin: new paragraph
        current.append(line)
    flush()
    return blocks


def load(path: Path) -> ParsedFile:
    title, meta, lines = split_header(drop_page_numbers(extract_pages(path)))
    return ParsedFile(title=title, meta=meta, blocks=lines_to_blocks(lines))
