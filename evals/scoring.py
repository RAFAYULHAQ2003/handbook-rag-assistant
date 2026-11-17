"""Score one answer against one eval question, following the rules in evals/README.md.

Pure functions with no model calls, so the scoring itself is tested and repeatable.
"""

import statistics
from collections import defaultdict

from evals.matching import contains_fact, facts_matched
from rag.answer import strip_markers
from rag.validity import State

STALE = (State.SUPERSEDED, State.DEPRECATED)


def _citation_passage_text(citation, passages_by_doc) -> str:
    """The passage a citation sits in (the smallest one covering it), so facts are checked in context."""
    covering = [p for p in passages_by_doc.get(citation.doc_id, [])
                if p.start <= citation.start < p.end]
    if not covering:
        return citation.quote
    return min(covering, key=lambda p: p.end - p.start).text


def asks_for_stale(question: dict, statuses) -> bool:
    """The question's gold documents include an outdated version, so citing it is legitimate."""
    return any(d in statuses and not statuses[d].in_force for d in question["expected_doc_ids"])


def score_answer(question: dict, answer, statuses, passages_by_doc) -> dict:
    """One row of results: did the answer pass, and how good were its citations."""
    expect = question["expect"]
    groups = question["must_include"]
    facts = facts_matched(strip_markers(answer.answer), groups) if groups else []  # "16 days [1]" -> "16 days"
    facts_ok = all(facts)
    verdict_ok = None if "verdict" not in question else answer.verdict == question["verdict"]
    errored = answer.error is not None and answer.answerable is None

    if errored:
        passed = False
    elif expect == "refuse":
        passed = answer.answerable is False
    elif expect == "clarify":
        passed = bool(answer.needs_clarification) or (bool(groups) and facts_ok)
    else:  # answer
        passed = answer.answerable is True and facts_ok and verdict_ok is not False

    # citations: does a cited passage carry an expected fact, and is it from a version that applies?
    stale_ok = asks_for_stale(question, statuses)
    passage_texts = [_citation_passage_text(c, passages_by_doc) for c in answer.citations]
    supports = [any(contains_fact(t, g) for g in groups) for t in passage_texts] if groups else []
    applies = [stale_ok or c.state not in STALE for c in answer.citations]
    citation_support = None if not groups or expect == "refuse" else any(supports)
    scored = bool(answer.citations) and bool(groups) and expect != "refuse"
    # strict: the cited passage itself carries an expected fact
    citation_precision = (sum(s and a for s, a in zip(supports, applies, strict=True)) / len(answer.citations)
                          if scored else None)
    # relevance (added after inspecting the first run, where strict precision penalised legitimate
    # reasoning steps such as "Tier 1: New York, San Francisco, Boston"): a fact, or a gold document
    gold = set(question["expected_doc_ids"]) | set(question.get("acceptable_doc_ids", []))
    if scored:
        relevant = [(s or c.doc_id in gold) and a for s, c, a in zip(supports, answer.citations, applies, strict=True)]
        citation_relevance = sum(relevant) / len(answer.citations)
    else:
        citation_relevance = None
    stale_only = bool(answer.citations) and not stale_ok and all(c.state in STALE for c in answer.citations)

    return {
        "id": question["id"],
        "category": question["category"],
        "expect": expect,
        "passed": passed,
        "facts": facts,
        "facts_ok": facts_ok,
        "verdict_expected": question.get("verdict"),
        "verdict": answer.verdict,
        "verdict_ok": verdict_ok,
        "answerable": answer.answerable,
        "needs_clarification": answer.needs_clarification,
        "false_refusal": expect != "refuse" and answer.answerable is False,
        "citation_support": citation_support,
        "citation_precision": citation_precision,
        "citation_relevance": citation_relevance,
        "citations": len(answer.citations),
        "has_markers": bool(answer.citations) and answer.answer != strip_markers(answer.answer),  # [n] in the text
        "dropped_citations": answer.dropped_citations,
        "unsupported": answer.unsupported,
        "stale_only": stale_only,
        "error": answer.error,
        "cached": answer.cached,
        "llm_calls": answer.llm_calls,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "cost_usd": answer.cost_usd,
        "latency_ms": answer.latency_ms,
        "context_docs": len(answer.retrieval.context) if answer.retrieval else 0,
        "answer": answer.answer,
        "cited": [{"doc_id": c.doc_id, "quote": c.quote} for c in answer.citations],
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarise(rows: list[dict]) -> dict:
    """Headline numbers for a run."""
    paid = [r for r in rows if r["llm_calls"] > 0]  # made a model call (not cached, not skipped)
    latencies = sorted(r["latency_ms"] for r in paid)
    by_expect = defaultdict(list)
    for r in rows:
        by_expect[r["expect"]].append(r["passed"])
    with_verdict = [r for r in rows if r["verdict_expected"] is not None]
    return {
        "questions": len(rows),
        "accuracy": _mean(float(r["passed"]) for r in rows),
        "accuracy_by_expect": {e: _mean(float(v) for v in vs) for e, vs in sorted(by_expect.items())},
        "verdict_accuracy": _mean(float(r["verdict_ok"]) for r in with_verdict) if with_verdict else None,
        "false_refusals": sum(r["false_refusal"] for r in rows),
        "citation_support": _mean(float(r["citation_support"]) if r["citation_support"] is not None else None
                                  for r in rows),
        "citation_precision": _mean(r["citation_precision"] for r in rows),
        "citation_relevance": _mean(r["citation_relevance"] for r in rows),
        # answers with sources whose text also ties claims to them with [n] markers
        "marker_coverage": _mean(float(r.get("has_markers", False)) for r in rows if r["citations"]),
        "unsupported_answers": sum(r["unsupported"] for r in rows),
        "dropped_quotes": sum(r["dropped_citations"] for r in rows),
        "stale_only_answers": sum(r["stale_only"] for r in rows),
        "errors": sum(r["error"] is not None and r["answerable"] is None for r in rows),
        "model_calls": sum(r["llm_calls"] for r in rows),
        "cached_answers": sum(r["cached"] for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "cost_usd_this_run": sum(r["cost_usd"] for r in paid),  # what this run actually spent
        "cost_usd_full_run": sum(r["cost_usd"] for r in rows),  # what all answers cost when first made
        # timed on paid calls only; cached answers return instantly and would flatter it
        "latency_ms_p50": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
    }


def by_category(rows: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r)
    return {c: {"n": len(rs), "accuracy": _mean(float(r["passed"]) for r in rs)} for c, rs in sorted(groups.items())}
