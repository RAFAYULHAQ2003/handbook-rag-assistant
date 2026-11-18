"""Excel workbooks (.xlsx): each sheet becomes a heading, each row a table row keyed by the header row."""

from pathlib import Path

from rag.loaders.base import LoaderError, ParsedFile, RawBlock, clean_inline, table_row_text

MAX_ROWS = 5000  # per sheet; a handbook table never needs more, and it bounds the work on a huge upload


def load(path: Path) -> ParsedFile:
    try:
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)  # cached values, not formulas
    except Exception as exc:  # not a zip, encrypted, corrupt
        raise LoaderError(f"cannot read Excel file: {exc}") from exc
    parsed = ParsedFile()
    try:
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True, max_row=MAX_ROWS):
                cells = [clean_inline("" if v is None else str(v)) for v in row]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            if len(workbook.worksheets) > 1:
                parsed.blocks.append(RawBlock("heading", clean_inline(sheet.title), level=1))
            headers = rows[0]
            width = max(len(r) for r in rows)
            headers = [h or f"column {i + 1}" for i, h in enumerate(headers + [""] * (width - len(headers)))]
            for cells in rows[1:]:
                pairs = [(h, c) for h, c in zip(headers, cells, strict=False) if c]
                if pairs:
                    parsed.blocks.append(RawBlock("table_row", table_row_text([h for h, _ in pairs], [c for _, c in pairs])))
    finally:
        workbook.close()
    return parsed
