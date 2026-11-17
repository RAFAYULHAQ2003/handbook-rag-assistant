import csv
from pathlib import Path

from rag.loaders.base import (
    LoaderError,
    ParsedFile,
    RawBlock,
    clean_inline,
    read_text_file,
    table_row_text,
)


def _dialect(sample: str):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel  # plain comma-separated


def load(path: Path) -> ParsedFile:
    text = read_text_file(path)
    try:
        rows = [[clean_inline(c) for c in row] for row in csv.reader(text.splitlines(), _dialect(text[:4096]))]
    except csv.Error as exc:
        raise LoaderError(f"cannot read CSV: {exc}") from exc

    rows = [r for r in rows if any(r)]
    parsed = ParsedFile()
    if not rows:
        return parsed
    headers = rows[0]
    width = max(len(r) for r in rows)
    headers = headers + [f"column {i + 1}" for i in range(len(headers), width)]  # never drop extra cells
    for cells in rows[1:]:
        pairs = [(h, c) for h, c in zip(headers, cells, strict=False) if c]
        parsed.blocks.append(RawBlock("table_row", table_row_text([h for h, _ in pairs], [c for _, c in pairs])))
    return parsed
