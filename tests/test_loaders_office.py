"""Uploads in Excel, PowerPoint and JSON."""

import json

import pytest

from rag.loaders import LoaderError, parse_file


def test_excel_rows_become_table_rows_keyed_by_the_header(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Rates"
    ws.append(["City", "Hotel cap"])
    ws.append(["Lisbon", 180])
    ws.append([None, None])
    ws2 = wb.create_sheet("Contacts")
    ws2.append(["Team", "Lead"])
    ws2.append(["Platform", "Maya Brooks"])
    path = tmp_path / "rates.xlsx"
    wb.save(path)
    blocks = parse_file(path).blocks
    assert [(b.kind, b.text) for b in blocks] == [
        ("heading", "Rates"), ("table_row", "City: Lisbon | Hotel cap: 180"),
        ("heading", "Contacts"), ("table_row", "Team: Platform | Lead: Maya Brooks")]


def test_powerpoint_slides_become_headings_with_their_text(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])  # title and content
    slide.shapes.title.text = "Offsite plan"
    body = slide.placeholders[1].text_frame
    body.text = "Lisbon in March"
    body.add_paragraph().text = "Flights via VoyageHub"
    body.paragraphs[1].level = 1
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(4), Inches(4), Inches(1)).table
    table.cell(0, 0).text, table.cell(0, 1).text = "Day", "Agenda"
    table.cell(1, 0).text, table.cell(1, 1).text = "1", "Kick-off"
    path = tmp_path / "plan.pptx"
    deck.save(path)
    blocks = [(b.kind, b.text) for b in parse_file(path).blocks]
    assert blocks[0] == ("heading", "Offsite plan")
    assert ("paragraph", "Lisbon in March") in blocks and ("list_item", "Flights via VoyageHub") in blocks
    assert ("table_row", "Day: 1 | Agenda: Kick-off") in blocks


def test_json_nesting_becomes_headings_values_and_rows(tmp_path):
    path = tmp_path / "team.json"
    path.write_text(json.dumps({"name": "Platform", "lead": "Maya Brooks",
                                "members": [{"name": "Ali", "role": "SRE"}], "tools": ["Grafana"]}))
    blocks = [(b.kind, b.text) for b in parse_file(path).blocks]
    assert ("paragraph", "name: Platform") in blocks and ("paragraph", "lead: Maya Brooks") in blocks
    assert ("table_row", "name: Ali | role: SRE") in blocks and ("list_item", "Grafana") in blocks


@pytest.mark.parametrize("name, data", [("bad.xlsx", b"not a zip"), ("bad.pptx", b"not a zip"), ("bad.json", b"{oops")])
def test_broken_files_are_clear_errors(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(LoaderError):
        parse_file(path)


def test_the_corpus_manifest_is_not_loaded_as_a_document(docs):
    assert "manifest" not in docs
