import re
import zipfile
from pathlib import Path

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag.loaders.base import (
    LoaderError,
    ParsedFile,
    RawBlock,
    clean_inline,
    is_header_line,
    parse_header_line,
    table_row_text,
)

_HEADING_STYLE = re.compile(r"^Heading (\d)$")


def _paragraph(p: Paragraph, parsed: ParsedFile) -> None:
    text = clean_inline(p.text)
    if not text:
        return
    style = p.style.name if p.style is not None else ""
    if style == "Title" and parsed.title is None:
        parsed.title = text
    elif is_header_line(text) and not parsed.meta:
        parsed.meta = parse_header_line(text)
    elif m := _HEADING_STYLE.match(style):
        parsed.blocks.append(RawBlock("heading", text, level=int(m.group(1))))
    elif style.startswith("List"):
        parsed.blocks.append(RawBlock("list_item", text))
    else:
        parsed.blocks.append(RawBlock("paragraph", text))


def _row_cells(row) -> list[str]:
    """Cell texts of a row; a merged cell is reported once, not once per grid column."""
    cells, seen = [], set()
    for cell in row.cells:
        if id(cell._tc) in seen:
            continue
        seen.add(id(cell._tc))
        cells.append(clean_inline(cell.text))
    return cells


def _table(t: Table, parsed: ParsedFile) -> None:
    rows = [_row_cells(row) for row in t.rows]
    if not rows:
        return
    headers, body = rows[0], rows[1:]
    if not body:  # a one-row table has no header row
        headers, body = [""] * len(rows[0]), rows
    for cells in body:
        parsed.blocks.append(RawBlock("table_row", table_row_text(headers, cells)))


def load(path: Path) -> ParsedFile:
    try:
        document = docx.Document(str(path))
    except (PackageNotFoundError, zipfile.BadZipFile) as exc:
        raise LoaderError("not a valid Word (.docx) file") from exc
    except Exception as exc:  # malformed XML inside a valid zip, and similar
        raise LoaderError(f"cannot read Word file: {exc}") from exc

    parsed = ParsedFile()
    try:
        for element in document.element.body.iterchildren():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "p":
                _paragraph(Paragraph(element, document), parsed)
            elif tag == "tbl":
                _table(Table(element, document), parsed)
    except Exception as exc:
        raise LoaderError(f"cannot read Word file: {exc}") from exc
    return parsed
