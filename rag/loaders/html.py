import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString

from rag.loaders.base import (
    LoaderError,
    ParsedFile,
    RawBlock,
    clean_inline,
    table_row_text,
)

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_UNITS = _HEADINGS | {"p", "li", "tr", "dt", "dd", "pre", "blockquote", "caption", "figcaption",
                      "div", "section", "article", "main", "header", "footer", "aside", "td", "th", "body"}
_DROP = ["script", "style", "noscript", "template", "head"]


_HIDDEN_STYLE = re.compile(r"(display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)


def _is_hidden(tag) -> bool:
    return tag.has_attr("hidden") or bool(_HIDDEN_STYLE.search(tag.get("style", "")))


def _unit(string: NavigableString):
    """The element a piece of text belongs to: its nearest heading, list item, row or block."""
    for parent in string.parents:
        if parent.name in {"li", "tr"} | _HEADINGS:
            return parent
    return string.find_parent(list(_UNITS))


def load(path: Path) -> ParsedFile:
    try:
        soup = BeautifulSoup(path.read_bytes(), "lxml")
    except Exception as exc:
        raise LoaderError(f"cannot read HTML: {exc}") from exc

    parsed = ParsedFile()
    if soup.title and soup.title.string:
        parsed.title = clean_inline(soup.title.string)
    for tag in soup(_DROP):
        tag.decompose()
    # text a reader never sees is not part of the document (and is a classic place to hide instructions)
    for tag in soup.find_all(_is_hidden):
        tag.decompose()

    # group text by its unit, keeping document order
    units: dict[int, tuple] = {}
    for string in soup.find_all(string=True):
        if isinstance(string, Comment) or not string.strip():
            continue
        unit = _unit(string)
        if unit is None:
            continue
        units.setdefault(id(unit), (unit, []))[1].append(str(string))

    table_headers: dict[int, list[str]] = {}
    for unit, _parts in units.values():
        if unit.name == "tr":
            cells = [clean_inline(c.get_text(" ")) for c in unit.find_all(["th", "td"])]
            table = id(unit.find_parent("table"))
            if unit.find("td") is None and table not in table_headers:
                table_headers[table] = cells
                continue
            headers = table_headers.get(table, [""] * len(cells))
            parsed.blocks.append(RawBlock("table_row", table_row_text(headers, cells)))
            continue
        text = clean_inline(" ".join(_parts))
        if unit.name in _HEADINGS:
            parsed.blocks.append(RawBlock("heading", text, level=int(unit.name[1])))
        elif unit.name == "li":
            parsed.blocks.append(RawBlock("list_item", text))
        else:
            parsed.blocks.append(RawBlock("paragraph", text))
    return parsed
