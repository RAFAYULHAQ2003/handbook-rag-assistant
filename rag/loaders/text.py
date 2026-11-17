import re
from pathlib import Path

from rag.loaders.base import ParsedFile, RawBlock, clean_inline, read_text_file

_LIST = re.compile(r"^(?:[-*•]|\d+[.)])\s+(.*)$")
_ENDS_SENTENCE = (".", ":", "!", "?", ";")


def _logical_lines(raw: list[str]) -> list[str | None]:
    """Merge hard-wrapped lines; None marks a blank line."""
    out: list[str | None] = []
    for line in raw:
        if not line:
            out.append(None)
            continue
        prev = out[-1] if out else None
        if (prev is not None and not prev.endswith(_ENDS_SENTENCE)
                and line[0].islower() and not _LIST.match(line)):
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)
    return out


def _is_heading(line: str) -> bool:
    return len(line) <= 60 and not line.endswith(_ENDS_SENTENCE + (",",)) and not _LIST.match(line)


def load(path: Path) -> ParsedFile:
    lines = _logical_lines([line.strip() for line in read_text_file(path).splitlines()])
    parsed = ParsedFile()
    for i, line in enumerate(lines):
        if line is None:
            continue
        has_more = any(lines[i + 1:])
        if m := _LIST.match(line):
            parsed.blocks.append(RawBlock("list_item", clean_inline(m.group(1))))
        elif has_more and _is_heading(line):
            parsed.blocks.append(RawBlock("heading", clean_inline(line), level=1))
        else:
            parsed.blocks.append(RawBlock("paragraph", clean_inline(line)))
    return parsed
