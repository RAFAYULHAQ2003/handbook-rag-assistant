import pytest

from evals.matching import contains_fact
from rag.chunking import MAX_CHARS, TABLE_MAX_CHARS, chunk_document, chunk_documents
from rag.loaders import RawBlock, build_text
from rag.models import Document, DocumentMeta
from tests.test_eval_set import ALL


def make_doc(raw_blocks, doc_id="doc", title="Doc"):
    text, blocks = build_text(raw_blocks, title=title)
    meta = DocumentMeta(doc_id=doc_id, title=title, path=f"{doc_id}.md", format="md")
    return Document(meta=meta, text=text, blocks=blocks)


@pytest.fixture(scope="module")
def passages(docs):
    return chunk_documents(list(docs.values()))


def of(passages, doc_id, kind=None):
    return [p for p in passages if p.doc_id == doc_id and (kind is None or p.kind == kind)]


# ---------- invariants over the whole corpus ----------

def test_text_is_exactly_the_document_span(docs, passages):
    for p in passages:
        assert docs[p.doc_id].text[p.start:p.end] == p.text, p.passage_id
        assert p.text == p.text.strip()


def test_ids_unique_and_stable(docs, passages):
    assert len({p.passage_id for p in passages}) == len(passages)
    again = chunk_documents(list(docs.values()))
    assert [p.passage_id for p in again] == [p.passage_id for p in passages]


def test_no_text_is_lost(docs, passages):
    """Every non-heading block sits inside some text or row passage."""
    for doc in docs.values():
        spans = [(p.start, p.end) for p in of(passages, doc.doc_id) if p.kind in ("text", "row")]
        for b in doc.blocks:
            if b.kind != "heading":
                assert any(s <= b.start and b.end <= e for s, e in spans), (doc.doc_id, b.text)


def test_sizes_respect_limits(passages):
    for p in passages:
        limit = TABLE_MAX_CHARS if p.kind == "table" else MAX_CHARS
        assert len(p.text) <= limit, p.passage_id


def test_search_text_has_title_and_section(docs, passages):
    (p,) = [p for p in of(passages, "hr-leave-policy-v2") if "16 days" in p.text]
    assert p.search_text.startswith("Leave Policy v2 > 2. Annual leave\n")
    assert p.search_text.endswith(p.text)


def test_headings_are_context_not_passage_text(passages):
    assert not any(p.text.startswith("2. Annual leave") for p in of(passages, "hr-leave-policy-v2"))


# ---------- structure ----------

def test_one_passage_per_section(passages):
    sections = [p.section for p in of(passages, "hr-leave-policy-v2", "text")]
    assert len(sections) == len(set(sections)) == 7  # lead paragraph + six numbered sections


def test_word_table_rows_and_whole_table(passages):
    rows = of(passages, "hr-travel-policy", "row")
    assert [r.text for r in rows][0] == "City tier: Tier 1 | Maximum room rate per night: $275"
    assert all(r.section == ("4. Limits", "4.2 Hotels and daily spending limits") for r in rows)
    (table,) = of(passages, "hr-travel-policy", "table")
    assert table.text.startswith("Hotel room rates, excluding taxes, are capped per night by city tier:")
    assert table.text.endswith("$150")


def test_csv_rows_and_whole_table(passages):
    assert len(of(passages, "fin-per-diem-table", "row")) == 15
    (table,) = of(passages, "fin-per-diem-table", "table")
    assert table.text.count("\n") == 14
    (release_table,) = of(passages, "eng-release-calendar", "table")
    assert all(r in release_table.text for r in ["2026.21", "2026.22", "2026.23", "2026.24", "2026.25"])


def test_markdown_table_rows(passages):
    rows = of(passages, "eng-disaster-recovery", "row")
    assert len(rows) == 4
    (table,) = of(passages, "eng-disaster-recovery", "table")
    assert table.text.startswith("Recovery objectives are set per service")


# ---------- splitting ----------

def test_long_section_splits_at_block_boundaries():
    paras = [RawBlock("paragraph", f"Paragraph {i} " + "x" * 90 + ".") for i in range(30)]
    doc = make_doc([RawBlock("heading", "Rules", level=1), *paras])
    ps = chunk_document(doc)
    assert len(ps) > 1
    assert all(len(p.text) <= MAX_CHARS for p in ps)
    assert all(p.section == ("Rules",) for p in ps)
    assert "\n".join(p.text for p in ps) == doc.text[doc.blocks[1].start:]  # nothing lost, no overlap


def test_over_long_paragraph_splits_at_sentences():
    sentence = "Employees must follow this rule at all times and report problems. "
    doc = make_doc([RawBlock("paragraph", (sentence * 40).strip())])
    ps = chunk_document(doc)
    assert len(ps) > 1
    assert all(len(p.text) <= MAX_CHARS for p in ps)
    assert all(p.text.endswith(".") for p in ps)  # never cut mid-sentence
    assert " ".join(p.text for p in ps) == doc.text


def test_single_huge_sentence_kept_whole():
    doc = make_doc([RawBlock("paragraph", "word " * 400 + "end.")])
    (p,) = chunk_document(doc)
    assert p.text == doc.text


def test_large_table_becomes_groups_of_rows():
    rows = [RawBlock("table_row", f"item: Widget {i} | price: ${i}.00 | note: " + "y" * 60) for i in range(80)]
    doc = make_doc(rows)
    ps = chunk_document(doc)
    tables = [p for p in ps if p.kind == "table"]
    assert len([p for p in ps if p.kind == "row"]) == 80
    assert len(tables) > 1 and all(len(t.text) <= TABLE_MAX_CHARS for t in tables)


def test_sections_are_not_merged():
    doc = make_doc([RawBlock("heading", "A", level=1), RawBlock("paragraph", "Alpha."),
                    RawBlock("heading", "B", level=1), RawBlock("paragraph", "Beta.")])
    assert [(p.section, p.text) for p in chunk_document(doc)] == [(("A",), "Alpha."), (("B",), "Beta.")]


def test_heading_only_document_has_no_passages():
    assert chunk_document(make_doc([RawBlock("heading", "Empty section", level=1)])) == []


# ---------- facts stay whole ----------

@pytest.mark.parametrize("source,q", ALL, ids=[q["id"] for _, q in ALL])
def test_each_expected_fact_fits_in_one_passage(passages, source, q):
    """A fact split across two passages could never be cited from one; check none is."""
    candidates = [p for p in passages if p.doc_id in q["expected_doc_ids"]]
    for group in q["must_include"]:
        assert any(contains_fact(p.text, group) for p in candidates), (q["id"], group)


def test_every_passage_lies_within_one_section(docs, passages):
    for p in passages:
        inside = [b for b in docs[p.doc_id].blocks if p.start <= b.start and b.end <= p.end]
        assert inside and {b.section for b in inside} == {p.section}, p.passage_id
