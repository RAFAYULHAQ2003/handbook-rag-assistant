"""JSON files: nested keys become headings, scalar values become "key: value" lines, lists of
objects become table rows."""

import json
from pathlib import Path

from rag.loaders.base import LoaderError, ParsedFile, RawBlock, clean_inline, read_text_file, table_row_text

MAX_BLOCKS = 20000


def _walk(value, name: str, level: int, out: list[RawBlock]) -> None:
    if len(out) >= MAX_BLOCKS:
        return
    if isinstance(value, dict):
        if name:
            out.append(RawBlock("heading", clean_inline(name), level=min(level, 6)))
        scalars = [(k, v) for k, v in value.items() if not isinstance(v, (dict, list))]
        for k, v in scalars:
            out.append(RawBlock("paragraph", clean_inline(f"{k}: {v}")))
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                _walk(v, str(k), level + 1, out)
    elif isinstance(value, list):
        if name:
            out.append(RawBlock("heading", clean_inline(name), level=min(level, 6)))
        for item in value:
            if isinstance(item, dict) and all(not isinstance(v, (dict, list)) for v in item.values()):
                keys = [str(k) for k, v in item.items() if v not in (None, "")]
                vals = [clean_inline(str(v)) for v in item.values() if v not in (None, "")]
                if keys:
                    out.append(RawBlock("table_row", table_row_text(keys, vals)))
            elif isinstance(item, (dict, list)):
                _walk(item, "", level + 1, out)
            elif item not in (None, ""):
                out.append(RawBlock("list_item", clean_inline(str(item))))
    elif value not in (None, ""):
        out.append(RawBlock("paragraph", clean_inline(f"{name}: {value}" if name else str(value))))


def load(path: Path) -> ParsedFile:
    try:
        data = json.loads(read_text_file(path))
    except json.JSONDecodeError as exc:
        raise LoaderError(f"not valid JSON: {exc}") from exc
    parsed = ParsedFile()
    _walk(data, "", 1, parsed.blocks)
    return parsed
