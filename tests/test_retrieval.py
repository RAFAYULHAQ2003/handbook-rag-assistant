"""Search: tokenising, BM25, fusion, validity-aware ranking, context assembly, embedding cache."""

import hashlib
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from evals.questions import load_questions, reference_date_for
from rag.index import HandbookIndex
from rag.retrieval.bm25 import BM25
from rag.retrieval.embeddings import EmbeddingCache
from rag.retrieval.search import (
    MAX_PASSAGES_PER_DOC,
    ReferenceFinder,
    Retriever,
    build_context,
)
from rag.retrieval.text import tokenize
from rag.validity import State

TODAY = date(2026, 11, 17)


class FakeEmbedder:
    """Deterministic bag-of-words vectors: no model download, same answer every run."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def index(docs):
    return HandbookIndex.build(list(docs.values()), TODAY, mode="bm25")


@pytest.fixture(scope="module")
def hybrid_index(docs):
    return HandbookIndex.build(list(docs.values()), TODAY, mode="hybrid",
                               embeddings=EmbeddingCache(FakeEmbedder(), None))


def ranked_docs(index, question, ref=None):
    return list(dict.fromkeys(h.passage.doc_id for h in index.retrieve(question, ref).pool))


def context_ids(index, question, ref=None):
    return {c.doc_id: c for c in index.retrieve(question, ref).context}


# ---------- tokenising ----------

@pytest.mark.parametrize("text,expected", [
    ("Claims above $2,500", ["claim", "above", "2500"]),
    ("x-request-id header", ["x-request-id", "x", "request", "id", "header"]),
    ("SEV2 policies", ["sev2", "policy"]),
    ("What is the bonus?", ["bonus"]),
    ("access process", ["access", "process"]),  # "ss" endings are not plurals
])
def test_tokenize(text, expected):
    assert tokenize(text) == expected


# ---------- BM25 ----------

def test_bm25_ranks_the_matching_text_first():
    bm25 = BM25(["the referral bonus is $3,000", "laptops are refreshed after 3 years", "vpn sessions"])
    scores = bm25.scores("referral bonus")
    assert scores.index(max(scores)) == 0
    assert scores[1] == scores[2] == 0


def test_bm25_rare_terms_weigh_more():
    # "leave" repeated three times loses to one rare "parental": that is what IDF is for
    bm25 = BM25(["leave leave leave policy", "leave policy", "leave rules", "parental"])
    scores = bm25.scores("leave parental")
    assert scores.index(max(scores)) == 3


def test_bm25_empty_and_unknown_query():
    assert BM25(["text"]).scores("zzz") == [0.0]
    assert BM25([]).scores("anything") == []


# ---------- ranking on the real handbook ----------

def test_exact_terms_found_by_keyword_search(index):
    assert ranked_docs(index, "What paperwork does a new vendor need? W-9")[0] == "fin-vendor-policy"


def test_current_version_ranks_above_superseded(index):
    order = ranked_docs(index, "How many days of annual leave do I get?")
    assert order[0] == "hr-leave-policy-v2"
    assert order.index("hr-leave-policy-v2") < order.index("hr-leave-policy-v1")


def test_question_naming_an_old_version_is_not_penalised(index):
    assert "hr-leave-policy-v1" in ranked_docs(index, "How many leave days did Leave Policy v1 give?")[:3]
    assert "eng-deploy-runbook-2023" in ranked_docs(index, "What did the 2023 deploy runbook say about rollback?")[:3]


def test_stale_versions_drop_in_the_ranking(index):
    statuses = index.statuses(TODAY)
    q = "rollback window after a deploy"
    rank = {}
    for name, states in (("states", statuses), ("text", None)):
        pool = index.retriever.search(q, k=8, statuses=states).pool
        rank[name] = next(h.rank for h in pool if h.passage.doc_id == "eng-deploy-runbook-2023")
    assert rank["states"] > rank["text"]


def test_meeting_notes_are_not_ranked_lower(index):
    # the canary decision lives only in meeting notes; the answer step sees that they are informal
    assert ranked_docs(index, "Was the proposal to shorten the canary stage to five minutes accepted?")[0] == \
        "notes-platform-2026-03"


def test_at_most_two_passages_per_document(index):
    hits = index.retrieve("What is the per-diem for an L1 employee visiting the Denver office?").hits
    per_doc = {}
    for h in hits:
        per_doc[h.passage.doc_id] = per_doc.get(h.passage.doc_id, 0) + 1
    assert max(per_doc.values()) <= MAX_PASSAGES_PER_DOC
    assert len(hits) == 8 and len(per_doc) >= 4


def test_denver_question_gets_the_tier_rule(index):
    # the per-diem table alone cannot say Denver is Tier 3; the Travel Policy must come too
    assert "hr-travel-policy" in context_ids(index, "What is the per-diem for an L1 employee visiting the Denver office?")


def test_states_follow_the_request_date_without_a_rebuild(docs):
    live = HandbookIndex.build(list(docs.values()), mode="bm25")  # no fixed date
    before = live.statuses("2026-12-31")["hr-parental-leave-policy-2027"].state
    after = live.statuses("2027-01-01")["hr-parental-leave-policy-2027"].state
    assert (before, after) == (State.NOT_YET_EFFECTIVE, State.IN_FORCE)
    for day, expected in (("2026-12-31", State.NOT_YET_EFFECTIVE), ("2027-01-01", State.IN_FORCE)):
        result = live.retrieve("How much leave does a secondary caregiver get?", day)
        assert result.reference_date == date.fromisoformat(day)
        revision = next(c for c in result.context if c.doc_id == "hr-parental-leave-policy-2027")
        assert revision.state is expected


def test_default_date_is_today_in_the_company_time_zone(docs, monkeypatch):
    monkeypatch.delenv("REFERENCE_DATE", raising=False)
    live = HandbookIndex.build(list(docs.values()), mode="bm25")
    assert live.retrieve("leave").reference_date == datetime.now(ZoneInfo("America/Chicago")).date()


def test_hybrid_fuses_both_rankings(hybrid_index):
    result = hybrid_index.retrieve("referral bonus amount", k=3)
    assert result.hits[0].passage.doc_id == "hr-referral-policy"
    assert result.hits[0].score >= result.hits[1].score >= result.hits[2].score
    assert result.top_bm25 is not None and result.top_cosine is not None


def test_relevance_hints_returned(index):
    result = index.retrieve("referral bonus")
    assert result.top_bm25 > 0 and result.top_cosine is None  # a keyword-only index has no cosine


def test_blank_question_returns_nothing(index):
    result = index.retrieve("   ")
    assert result.hits == [] and result.context == []


def test_modes_validated():
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        Retriever([], [], mode="magic")
    with pytest.raises(ValueError, match="needs an embedding model"):
        Retriever([], [], mode="hybrid")


def test_falls_back_to_keyword_search_without_a_model(docs, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("no network")
    monkeypatch.setattr("rag.index.load_embeddings", broken)
    assert HandbookIndex.build(list(docs.values()), TODAY, mode="hybrid").mode == "bm25"


# ---------- context assembly ----------

def test_small_documents_are_sent_whole(index):
    travel = context_ids(index, "What is the hotel limit in San Francisco?")["hr-travel-policy"]
    assert travel.whole_document
    text = travel.text(index.docs["hr-travel-policy"])
    assert "Tier 1: New York, San Francisco, Boston" in text and "$275" in text


def test_overlapping_passages_are_merged(index):
    context = index.retrieve("hotel room rate per night tier 1 tier 2 tier 3").context
    for c in context:
        assert all(a[1] < b[0] for a, b in zip(c.spans, c.spans[1:], strict=False)), c.doc_id
    assert len({c.doc_id for c in context}) == len(context)


def test_large_document_sends_only_matching_spans(index):
    runbook = context_ids(index, "What happens if the primary on-call does not acknowledge a page?")["eng-oncall-runbook"]
    assert not runbook.whole_document
    assert "15 minutes" in runbook.text(index.docs["eng-oncall-runbook"])


def test_stale_document_brings_its_current_version(index):
    by_id = context_ids(index, "What did the 2023 deploy runbook say about the rollback window?")
    assert by_id["eng-deploy-runbook-2023"].state is State.DEPRECATED
    assert "eng-deploy-runbook" in by_id
    assert "Use Production Deploy Runbook" in by_id["eng-deploy-runbook-2023"].status_note


def test_related_large_document_sends_its_best_matching_passage(index):
    statuses = index.statuses(TODAY)
    pool = index.retriever.search("primary on-call does not acknowledge a page escalation", k=8,
                                  statuses=statuses).pool
    runbook_hit = next(h for h in pool if h.passage.doc_id == "eng-oncall-runbook")
    other = next(h for h in pool if h.passage.doc_id != "eng-oncall-runbook")
    # make the runbook arrive only as a related document: its best-ranked passage must be the one sent
    faked = {**statuses, other.passage.doc_id: replace(statuses[other.passage.doc_id],
                                                       upcoming=("eng-oncall-runbook",))}
    context = build_context([other], index.docs, faked, pool, index.passages_by_doc)
    added = next(c for c in context if c.doc_id == "eng-oncall-runbook")
    assert added.spans == [(runbook_hit.passage.start, runbook_hit.passage.end)]


def test_current_policy_brings_its_upcoming_revision(index):
    by_id = context_ids(index, "How much leave does a secondary caregiver get?")
    assert "hr-parental-leave-policy" in by_id and "hr-parental-leave-policy-2027" in by_id
    assert by_id["hr-parental-leave-policy-2027"].state is State.NOT_YET_EFFECTIVE


def test_referenced_documents_are_added(index):
    assert "hr-probation-policy" in context_ids(index, "Can I claim the home office allowance two months after joining?")


def test_references_come_only_from_the_top_passages(index):
    statuses = index.statuses(TODAY)
    hits = index.retriever.search("home office allowance after joining", k=8, statuses=statuses).hits
    none = build_context(hits, index.docs, statuses, references=index.references, reference_hits=0)
    some = build_context(hits, index.docs, statuses, references=index.references, reference_hits=3)
    assert not any(c.reason.startswith("referenced by") for c in none)
    assert any(c.reason.startswith("referenced by") for c in some)


def test_references_never_add_stale_documents(index):
    context = index.retrieve("How many days of annual leave do I get per year?", k=1).context
    assert all(c.state is State.IN_FORCE for c in context if c.reason.startswith("referenced by"))


def test_context_respects_the_budget(index):
    statuses = index.statuses(TODAY)
    hits = index.retriever.search("policy", k=50, statuses=statuses, max_per_doc=50).hits
    context = build_context(hits, index.docs, statuses, budget=3000)
    assert sum(e - s for c in context for s, e in c.spans) <= 3000 or len(context) == 1


def test_top_document_kept_even_if_over_budget(index):
    statuses = index.statuses(TODAY)
    hits = index.retriever.search("on-call escalation severity", k=3, statuses=statuses).hits
    context = build_context(hits, index.docs, statuses, budget=10)
    assert context and context[0].doc_id == hits[0].passage.doc_id


def test_reference_finder(docs):
    finder = ReferenceFinder(list(docs.values()))
    assert finder.find("after the introductory period") == ["hr-probation-policy"]
    assert finder.find("See Email and Messaging Guidelines.") == ["it-email-messaging"]
    assert finder.find("limits in section 4.2 of the Travel Policy") == ["hr-travel-policy"]
    assert finder.find("business travel within limits") == []  # "travel" alone never links
    assert finder.find("see the VPN Guide and the Travel Policy") == ["it-vpn-guide", "hr-travel-policy"]


# ---------- embedding cache ----------

def test_embedding_cache_embeds_each_text_once(tmp_path):
    embedder = FakeEmbedder()
    path = tmp_path / "vectors.npz"
    first = EmbeddingCache(embedder, path).get(["alpha", "beta"])
    assert embedder.calls == 2
    again = EmbeddingCache(embedder, path).get(["beta", "alpha", "gamma"])
    assert embedder.calls == 3  # only "gamma" was new
    np.testing.assert_allclose(again[0], first[1])
    assert not list(tmp_path.glob("*.tmp*"))  # atomic write left no temp file


def test_caches_saving_at_once_never_corrupt_the_file(tmp_path):
    import threading
    path = tmp_path / "vectors.npz"
    caches = [EmbeddingCache(FakeEmbedder(), path) for _ in range(4)]
    threads = [threading.Thread(target=c.get, args=([f"text {i} {j}" for j in range(20)],))
               for i, c in enumerate(caches)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    reloaded = EmbeddingCache(FakeEmbedder(), path)  # must load cleanly
    assert len(reloaded._vectors) >= 20
    assert not list(tmp_path.glob("*.tmp*"))


# ---------- quality gates on the dev set ----------

def _context_recall(index, questions):
    found = total = 0
    for q in questions:
        context = {c.doc_id for c in index.retrieve(q["question"], reference_date_for(q)).context}
        found += len(set(q["expected_doc_ids"]) & context)
        total += len(q["expected_doc_ids"])
    return found / total


def test_keyword_search_quality_gate(docs):
    """Guards against a change that silently makes retrieval worse (dev set; holdout stays blind)."""
    questions = [q for q in load_questions("dev") if q["expected_doc_ids"]]
    assert _context_recall(HandbookIndex.build(list(docs.values()), mode="bm25"), questions) >= 0.97


def _model_downloaded() -> bool:
    from rag.retrieval.embeddings import DEFAULT_CACHE_DIR
    return any((DEFAULT_CACHE_DIR / "models").glob("*bge-small-en-v1.5*"))


@pytest.mark.skipif(not _model_downloaded(), reason="embedding model not downloaded; run the app once to fetch it")
def test_hybrid_search_quality_gate(docs, monkeypatch):
    """The chosen configuration keeps its measured dev-set quality."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # never download during tests
    questions = [q for q in load_questions("dev") if q["expected_doc_ids"]]
    index = HandbookIndex.build(list(docs.values()), mode="hybrid")
    assert index.mode == "hybrid"
    hit1 = []
    for q in questions:
        hits = index.retrieve(q["question"], reference_date_for(q)).hits
        good = set(q["expected_doc_ids"]) | set(q.get("acceptable_doc_ids", []))
        hit1.append(bool(hits) and hits[0].passage.doc_id in good)
    assert sum(hit1) / len(hit1) >= 0.90
    assert _context_recall(index, questions) >= 0.98


def test_save_never_touches_another_workers_temp_file(tmp_path):
    path = tmp_path / "vectors.npz"
    other = path.with_suffix(".tmp.npz")  # another worker's half-written file
    other.write_bytes(b"in progress")
    EmbeddingCache(FakeEmbedder(), path).get(["alpha"])
    assert other.read_bytes() == b"in progress"
    assert len(EmbeddingCache(FakeEmbedder(), path)._vectors) == 1


# ---------- dated schedules for date questions ----------

def test_deploy_today_brings_in_the_release_calendar(index):
    """The runbook says a freeze starts 72 hours before every scheduled release but never names the
    calendar; a question about today needs the calendar's dates to answer yes or no."""
    context = {c.doc_id: c for c in index.retrieve("Can I deploy to production today?").context}
    assert "eng-deploy-runbook" in context and "eng-release-calendar" in context
    assert context["eng-release-calendar"].reason == "dated schedule for a question about a date"


def test_schedules_are_added_only_for_date_questions_about_them(index):
    assert "eng-release-calendar" not in {c.doc_id for c in index.retrieve("How do I deploy to production?").context}
    assert "eng-release-calendar" not in {
        c.doc_id for c in index.retrieve("How many days of annual leave do I get this year?").context}


def test_schedule_detection(index):
    assert set(index.schedules) == {"eng-release-calendar"}
    assert index.schedules["eng-release-calendar"].search("every scheduled release")
