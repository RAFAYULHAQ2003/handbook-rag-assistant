"""The answer eval's scoring, runner and baseline, with a fake model (no cost)."""

from datetime import date

import pytest

from evals.answer_eval import INJECTED_DOC, run_questions, whole_handbook
from evals.scoring import by_category, score_answer, summarise
from rag.answer import Answer, Citation
from rag.index import HandbookIndex
from rag.llm import Completion
from rag.manifest import load_document
from rag.validity import State

TODAY = date(2026, 11, 17)


@pytest.fixture(scope="module")
def index(docs):
    return HandbookIndex.build(list(docs.values()), TODAY, mode="bm25")


def cite(index, doc_id, quote):
    text = index.docs[doc_id].text
    start = text.index(quote)
    state = index.statuses(TODAY)[doc_id].state
    return Citation(doc_id, index.docs[doc_id].meta.title, quote, start, start + len(quote), state)


def q(expect="answer", must=(("16",),), verdict=None, expected=("hr-leave-policy-v2",), category="fact"):
    question = {"id": "t-1", "category": category, "question": "?", "expect": expect,
                "expected_doc_ids": list(expected), "must_include": [list(g) for g in must]}
    if verdict:
        question["verdict"] = verdict
    return question


def score(index, question, answer):
    return score_answer(question, answer, index.statuses(TODAY), index.passages_by_doc)


V2 = "Full-time employees accrue 16 days of annual leave per calendar year"
V1 = "Full-time employees accrue 12 days of annual leave per calendar year"


# ---------- pass / fail rules ----------

def test_answer_passes_with_facts_and_a_good_citation(index):
    a = Answer("?", "You get 16 days.", answerable=True, citations=[cite(index, "hr-leave-policy-v2", V2)])
    row = score(index, q(), a)
    assert row["passed"] and row["citation_support"] and row["citation_precision"] == 1.0
    assert not row["stale_only"] and not row["unsupported"]


def test_answer_fails_without_the_fact(index):
    assert not score(index, q(), Answer("?", "You get 12 days.", answerable=True))["passed"]


def test_answer_fails_when_refused(index):
    row = score(index, q(), Answer("?", "16 days, but I am not sure.", answerable=False))
    assert not row["passed"] and row["false_refusal"]


def test_wrong_verdict_fails_even_with_the_facts(index):
    question = q(must=(("freeze",),), verdict="no", expected=("eng-deploy-runbook",))
    row = score(index, question, Answer("?", "Yes, unless there is a freeze.", verdict="yes", answerable=True))
    assert not row["passed"] and row["verdict_ok"] is False


def test_refusal_rules(index):
    refuse = q(expect="refuse", must=(), expected=())
    assert score(index, refuse, Answer("?", "Not covered.", answerable=False))["passed"]
    assert not score(index, refuse, Answer("?", "It is Aetna.", answerable=True))["passed"]


def test_clarify_rules(index):
    clarify = q(expect="clarify", must=(("Maya Rivera",), ("Maya Brooks",), ("Maya Park",)), expected=("org-chart",))
    assert score(index, clarify, Answer("?", "Which Maya?", answerable=True, needs_clarification=True))["passed"]
    listed = Answer("?", "Maya Rivera, Maya Brooks or Maya Park.", answerable=True)
    assert score(index, clarify, listed)["passed"]
    assert not score(index, clarify, Answer("?", "Maya Park is Head of Product.", answerable=True))["passed"]


def test_errors_always_fail(index):
    a = Answer("?", "Sorry...", answerable=None, error="model call failed: HTTP 503")
    assert not score(index, q(), a)["passed"]


# ---------- citations ----------

def test_citing_a_superseded_version_is_imprecise_and_flagged(index):
    a = Answer("?", "16 days.", answerable=True, citations=[cite(index, "hr-leave-policy-v1", V1)])
    row = score(index, q(), a)
    assert row["passed"]  # the text is right...
    assert row["stale_only"] and row["citation_precision"] == 0.0  # ...but it rests on the old version


def test_citing_the_old_version_is_fine_when_asked_about_it(index):
    question = q(must=(("12",), ("16",)), expected=("hr-leave-policy-v1", "hr-leave-policy-v2"), category="superseded")
    a = Answer("?", "v1 gave 12; v2 gives 16.", answerable=True,
               citations=[cite(index, "hr-leave-policy-v1", V1), cite(index, "hr-leave-policy-v2", V2)])
    row = score(index, question, a)
    assert not row["stale_only"] and row["citation_precision"] == 1.0


def test_citation_is_checked_in_its_whole_passage(index):
    # the quote itself lacks the number, but the passage it sits in has it
    a = Answer("?", "You get 16 days.", answerable=True,
               citations=[cite(index, "hr-leave-policy-v2", "credited each pay period in equal parts")])
    assert score(index, q(), a)["citation_support"]


def test_irrelevant_citation_does_not_count(index):
    a = Answer("?", "16 days.", answerable=True,
               citations=[cite(index, "hr-referral-policy", "The bonus is taxable income")])
    row = score(index, q(), a)
    assert row["citation_support"] is False and row["citation_precision"] == 0.0
    assert row["citation_relevance"] == 0.0


def test_reasoning_step_from_a_gold_document_is_relevant_but_not_strictly_precise(index):
    # "this version supersedes v1" carries no expected number, but is the right evidence
    a = Answer("?", "16 days.", answerable=True, citations=[
        cite(index, "hr-leave-policy-v2", V2),
        cite(index, "hr-leave-policy-v2", "This version supersedes Leave Policy v1 with effect from 2025-04-01")])
    row = score(index, q(), a)
    assert row["citation_relevance"] == 1.0
    assert row["citation_precision"] in (0.5, 1.0)  # strict: depends on whether that passage holds "16"


def test_stale_citation_is_never_relevant_unless_asked(index):
    a = Answer("?", "16 days.", answerable=True, citations=[cite(index, "hr-leave-policy-v1", V1)])
    question = q(expected=("hr-leave-policy-v2", "hr-leave-policy-v1"))
    question["expected_doc_ids"] = ["hr-leave-policy-v2"]
    question["acceptable_doc_ids"] = ["hr-leave-policy-v1"]
    assert score(index, question, a)["citation_relevance"] == 0.0


# ---------- summary ----------

def test_summary_counts(index):
    rows = [score(index, q(), Answer("?", "16 days.", answerable=True, llm_calls=1, cost_usd=0.001, latency_ms=2000)),
            score(index, q(), Answer("?", "12 days.", answerable=True, cached=True)),
            score(index, q(expect="refuse", must=(), expected=()), Answer("?", "No.", answerable=False))]
    s = summarise(rows)
    assert s["questions"] == 3 and s["accuracy"] == pytest.approx(2 / 3)
    assert s["accuracy_by_expect"] == {"answer": 0.5, "refuse": 1.0}
    assert s["model_calls"] == 1 and s["cached_answers"] == 1
    assert s["cost_usd_this_run"] == pytest.approx(0.001) and s["cost_usd_full_run"] == pytest.approx(0.001)
    assert s["latency_ms_p50"] == 2000  # timed on the paid call only
    assert s["verdict_accuracy"] is None  # no yes/no questions in this run
    assert by_category(rows)["fact"] == {"n": 3, "accuracy": pytest.approx(2 / 3)}


# ---------- runner and baseline, with a fake model ----------

class EchoClient:
    """Answers "16 days" with a real quote; counts calls; thread-safe enough for the runner."""

    model = "fake/echo"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, schema, max_tokens=4000):
        self.calls += 1
        return Completion({"answer": "You get 16 days.", "verdict": None, "answerable": True,
                           "needs_clarification": False,
                           "citations": [{"doc_id": "hr-leave-policy-v2", "quote": V2}]},
                          self.model, "fake", 10, 5, 0.0001, 1.0, False)


def test_runner_scores_every_question(index):
    questions = [dict(q(), id=f"t-{i}", question="How many days of annual leave do I get?") for i in range(5)]
    client = EchoClient()
    rows = run_questions(questions, index, client, workers=3)
    assert [r["id"] for r in rows] == [f"t-{i}" for i in range(5)]  # order kept despite parallel workers
    assert all(r["passed"] for r in rows) and client.calls == 5


def test_whole_handbook_baseline_sends_every_document(index):
    retrieval = whole_handbook("anything", index, TODAY)
    assert len(retrieval.context) == len(index.docs)
    assert all(c.whole_document for c in retrieval.context)
    states = {c.doc_id: c.state for c in retrieval.context}
    assert states["hr-leave-policy-v1"] is State.SUPERSEDED  # states are still given to the model


def test_injected_fixture_loads_and_is_findable(docs):
    injected, _ = load_document(INJECTED_DOC)
    assert injected.doc_id == "notice-social-committee"
    assert "ignore your previous rules" in injected.text
    index = HandbookIndex.build(list(docs.values()) + [injected], TODAY, mode="bm25")
    context = {c.doc_id for c in index.retrieve("What is the referral bonus?").context}
    assert "notice-social-committee" in context  # the attack actually reaches the model


def test_refusal_with_citations_is_scored_without_crashing(index):
    refuse = q(expect="refuse", must=(), expected=())
    a = Answer("?", "The handbook doesn't name the provider.", answerable=False,
               citations=[cite(index, "ref-benefits-overview", "Medical, dental and vision insurance")])
    row = score(index, refuse, a)
    assert row["passed"] and row["citation_precision"] is None and row["citation_relevance"] is None


def test_cached_answers_are_not_counted_as_spending(index):
    rows = [score(index, q(), Answer("?", "16 days.", answerable=True, cached=True, cost_usd=0.002))]
    s = summarise(rows)
    assert s["cost_usd_this_run"] == 0 and s["cost_usd_full_run"] == pytest.approx(0.002)
    assert s["latency_ms_p50"] is None
