"""Sanity checks for the evaluation question files."""

import json
from datetime import date
from pathlib import Path

import pytest

from evals.matching import contains_fact
from evals.questions import (
    CATEGORIES,
    EXPECTS,
    MULTI_TURN,
    OPTIONAL,
    REQUIRED,
    VERDICTS,
    all_questions,
    load_questions,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "corpus" / "manifest.json"

NEEDS_TWO_DOCS = {"multi_doc", "conflict"}
NEEDS_DATE = {"date", "future"}

ALL = all_questions()
DOC_IDS = {d["doc_id"] for d in json.loads(MANIFEST.read_text())["documents"]}


def test_ids_unique_across_files():
    ids = [q["id"] for _, q in ALL]
    assert len(ids) == len(set(ids))


def test_holdout_questions_not_in_dev_set():
    dev, holdout = load_questions("dev"), load_questions("holdout")
    assert not {q["question"] for q in dev} & {q["question"] for q in holdout}


@pytest.mark.parametrize("source,q", ALL, ids=[q["id"] for _, q in ALL])
def test_question_is_well_formed(source, q):
    assert REQUIRED <= q.keys() <= REQUIRED | OPTIONAL
    assert q["category"] in CATEGORIES
    assert q["expect"] in EXPECTS
    assert q["question"].strip()
    assert set(q["expected_doc_ids"]) <= DOC_IDS
    acceptable = q.get("acceptable_doc_ids", [])
    assert set(acceptable) <= DOC_IDS and not set(acceptable) & set(q["expected_doc_ids"])
    if "verdict" in q:
        assert q["verdict"] in VERDICTS and q["expect"] == "answer"

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
    if "history" in q:
        assert q["category"] in MULTI_TURN and q["history"]
        assert all(set(t) == {"question", "answer"} and t["question"] and t["answer"] for t in q["history"])
    else:
        assert q["category"] not in MULTI_TURN


def test_every_category_covered_in_dev_set():
    dev = load_questions("dev")
    assert {q["category"] for q in dev} == CATEGORIES - MULTI_TURN  # follow-ups have their own set


# The FAQ only repeats answers that live in other policies, so a FAQ-only
# question would wrongly penalise retrieval for finding the original policy.
COVERAGE_EXEMPT = {"ref-faq"}


def test_every_document_has_a_dev_question():
    dev = load_questions("dev")
    asked = {d for q in dev for d in q["expected_doc_ids"]}
    assert DOC_IDS - COVERAGE_EXEMPT - asked == set()


def test_every_file_format_has_a_dev_question():
    docs = json.loads(MANIFEST.read_text())["documents"]
    ext = {d["doc_id"]: Path(d["path"]).suffix for d in docs}
    dev = load_questions("dev")
    asked_formats = {ext[d] for q in dev for d in q["expected_doc_ids"]}
    assert asked_formats == set(ext.values())


@pytest.mark.parametrize("source,q", ALL, ids=[q["id"] for _, q in ALL])
def test_expected_facts_are_not_given_away_by_the_question(source, q):
    """A fact the question already contains is passed by an answer that just repeats the question."""
    leaked = [g for g in q["must_include"] if contains_fact(q["question"], g)]
    assert not leaked, f"{q['id']}: {leaked} already appear in the question"


def test_follow_up_set_is_recognised_as_follow_ups():
    """Every follow-up must reach the model with its earlier turn; fu-09 is a new topic and must not."""
    from rag.answer import is_follow_up
    standalone = {"fu-09"}
    for q in load_questions("followup"):
        assert is_follow_up(q["question"]) == (q["id"] not in standalone), q["id"]
