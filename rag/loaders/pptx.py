"""PowerPoint decks (.pptx): each slide's title becomes a heading, its text paragraphs and tables follow."""

from pathlib import Path

from rag.loaders.base import LoaderError, ParsedFile, RawBlock, clean_inline, table_row_text


def load(path: Path) -> ParsedFile:
    try:
        from pptx import Presentation
        deck = Presentation(str(path))
    except Exception as exc:
        raise LoaderError(f"cannot read PowerPoint file: {exc}") from exc
    parsed = ParsedFile()
    for number, slide in enumerate(deck.slides, 1):
        title_shape = slide.shapes.title
        title = clean_inline(title_shape.text_frame.text) if title_shape is not None and title_shape.has_text_frame else ""
        parsed.blocks.append(RawBlock("heading", title or f"Slide {number}", level=1))
        for shape in slide.shapes:
            if shape is title_shape:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[clean_inline(c.text) for c in r.cells] for r in shape.table.rows]
                if rows:
                    headers = [h or f"column {i + 1}" for i, h in enumerate(rows[0])]
                    for cells in rows[1:]:
                        pairs = [(h, c) for h, c in zip(headers, cells, strict=False) if c]
                        if pairs:
                            parsed.blocks.append(RawBlock("table_row", table_row_text([h for h, _ in pairs], [c for _, c in pairs])))
            elif shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = clean_inline("".join(run.text for run in paragraph.runs))
                    if text:
                        parsed.blocks.append(RawBlock("list_item" if paragraph.level > 0 else "paragraph", text))
    return parsed
