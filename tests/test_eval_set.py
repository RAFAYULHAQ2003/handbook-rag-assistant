"""Sanity checks for the evaluation question files."""

import json
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILES = [ROOT / "evals" / "questions.jsonl", ROOT / "evals" / "questions_holdout.jsonl"]
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"

CATEGORIES = {
    "fact", "table", "superseded", "future", "deprecated", "multi_doc", "conflict",
    "date", "partial", "unanswerable", "ambiguous", "paraphrase", "injection",
}
NEEDS_TWO_DOCS = {"multi_doc", "conflict"}
NEEDS_DATE = {"date", "future"}
EXPECTS = {"answer", "refuse", "clarify"}
REQUIRED = {"id", "category", "question", "expect", "expected_doc_ids", "must_include"}


def load(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


ALL = [(p.name, q) for p in EVAL_FILES for q in load(p)]
DOC_IDS = {d["doc_id"] for d in json.loads(MANIFEST.read_text())["documents"]}


def test_ids_unique_across_files():
    ids = [q["id"] for _, q in ALL]
    assert len(ids) == len(set(ids))


def test_holdout_questions_not_in_dev_set():
    dev, holdout = (load(p) for p in EVAL_FILES)
    assert not {q["question"] for q in dev} & {q["question"] for q in holdout}


@pytest.mark.parametrize("source,q", ALL, ids=[q["id"] for _, q in ALL])
def test_question_is_well_formed(source, q):
    assert REQUIRED <= q.keys()
    assert q["category"] in CATEGORIES
    assert q["expect"] in EXPECTS
    assert q["question"].strip()
    assert set(q["expected_doc_ids"]) <= DOC_IDS

    for group in q["must_include"]:
        assert isinstance(group, list) and group and all(isinstance(s, str) and s for s in group)

    if q["expect"] == "refuse":
        assert not q["must_include"]
    else:
        assert q["must_include"] and q["expected_doc_ids"]

    if q["category"] in NEEDS_TWO_DOCS:
        assert len(q["expected_doc_ids"]) >= 2, "multi-document questions must name every document"
    if q["category"] in NEEDS_DATE:
        assert "reference_date" in q, "date-dependent questions must pin the date"
    if "reference_date" in q:
        date.fromisoformat(q["reference_date"])


def test_every_category_covered_in_dev_set():
    dev = load(EVAL_FILES[0])
    assert {q["category"] for q in dev} == CATEGORIES


# The FAQ only repeats answers that live in other policies, so a FAQ-only
# question would wrongly penalise retrieval for finding the original policy.
COVERAGE_EXEMPT = {"ref-faq"}


def test_every_document_has_a_dev_question():
    dev = load(EVAL_FILES[0])
    asked = {d for q in dev for d in q["expected_doc_ids"]}
    assert DOC_IDS - COVERAGE_EXEMPT - asked == set()


def test_every_file_format_has_a_dev_question():
    docs = json.loads(MANIFEST.read_text())["documents"]
    ext = {d["doc_id"]: Path(d["path"]).suffix for d in docs}
    dev = load(EVAL_FILES[0])
    asked_formats = {ext[d] for q in dev for d in q["expected_doc_ids"]}
    assert asked_formats == set(ext.values())
