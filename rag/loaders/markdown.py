import re
from pathlib import Path

import yaml

from rag.loaders.base import (
    LoaderError,
    ParsedFile,
    RawBlock,
    clean_inline,
    read_text_file,
    table_row_text,
)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}")
_FENCE = re.compile(r"^(```|~~~)")


def strip_markdown(text: str) -> str:
    """Plain text from inline Markdown: links, images, emphasis and code markers."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images -> alt text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> link text
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)  # bold
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)  # *italic*
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", text)  # _italic_
    text = text.replace("`", "")
    return clean_inline(text)


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise LoaderError(f"invalid frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        return {}, text
    meta = {str(k): str(v) for k, v in data.items() if v is not None}
    return meta, text[match.end():]


def _cells(line: str) -> list[str]:
    return [strip_markdown(c) for c in line.strip().strip("|").split("|")]


def load(path: Path) -> ParsedFile:
    meta, body = _frontmatter(read_text_file(path))
    parsed = ParsedFile(title=meta.pop("title", None), meta=meta)
    blocks = parsed.blocks

    paragraph: list[str] = []
    table_headers: list[str] | None = None
    in_fence = False

    def flush():
        if paragraph:
            blocks.append(RawBlock("paragraph", strip_markdown(" ".join(paragraph))))
            paragraph.clear()

    for line in body.splitlines():
        stripped = line.strip()

        # fenced code: kept as plain text, never parsed as headings or lists
        if _FENCE.match(stripped):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            if stripped:
                blocks.append(RawBlock("paragraph", clean_inline(stripped)))
            continue

        if stripped.startswith("|"):
            flush()
            if _TABLE_SEPARATOR.match(stripped):
                continue
            if table_headers is None:
                table_headers = _cells(stripped)
            else:
                blocks.append(RawBlock("table_row", table_row_text(table_headers, _cells(stripped))))
            continue
        table_headers = None

        if not stripped:
            flush()
        elif m := _HEADING.match(stripped):
            flush()
            blocks.append(RawBlock("heading", strip_markdown(m.group(2)), level=len(m.group(1))))
        elif m := _LIST.match(line):
            flush()
            blocks.append(RawBlock("list_item", strip_markdown(m.group(1))))
        elif line[:1].isspace() and not paragraph and blocks and blocks[-1].kind == "list_item":
            # indented continuation of the previous list item
            prev = blocks[-1]
            blocks[-1] = RawBlock("list_item", clean_inline(prev.text + " " + strip_markdown(stripped)))
        else:
            paragraph.append(stripped)
    flush()
    return parsed
