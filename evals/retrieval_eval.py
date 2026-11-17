"""Compare retrieval configurations on an eval set, with no language-model calls.

    python -m evals.retrieval_eval                 # dev set, all configurations
    python -m evals.retrieval_eval --set holdout   # only once tuning is finished
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path

from evals.questions import load_questions, reference_date_for
from rag.answer import is_follow_up
from rag.index import HandbookIndex
from rag.manifest import load_corpus
from rag.retrieval.embeddings import EmbeddingCache, FastEmbedder, cache_path_for

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
RESULTS_DIR = ROOT / "evals" / "results"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
BGE, MINILM = "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"
BASELINE = "bm25"


@dataclass(frozen=True)
class Config:
    name: str
    mode: str
    model: str | None = None
    prefix: str = ""
    use_validity: bool = True
    follow_references: bool = True


CONFIGS = [
    Config("bm25", "bm25"),
    Config("dense bge-small", "dense", BGE),
    Config("dense bge-small +prefix", "dense", BGE, BGE_PREFIX),
    Config("dense minilm", "dense", MINILM),
    Config("hybrid bge-small", "hybrid", BGE),
    Config("hybrid bge-small +prefix", "hybrid", BGE, BGE_PREFIX),
    Config("hybrid minilm", "hybrid", MINILM),
    Config("hybrid bge-small, no validity", "hybrid", BGE, use_validity=False),
    Config("hybrid bge-small, no references", "hybrid", BGE, follow_references=False),
    Config("bm25, no validity", "bm25", use_validity=False),
]


def score_question(q: dict, retrieval, statuses) -> dict:
    expected = set(q["expected_doc_ids"])
    good = expected | set(q.get("acceptable_doc_ids", []))
    ranked = list(dict.fromkeys(h.passage.doc_id for h in retrieval.pool))
    context = {c.doc_id for c in retrieval.context}
    first_good = next((i for i, d in enumerate(ranked) if d in good), None)
    # a stale version ranked first is harmless only when the question asks for that version
    asks_for_stale = any(not statuses[d].in_force for d in expected)
    top = ranked[0] if ranked else None
    stale_first = not asks_for_stale and top is not None and not statuses[top].in_force and \
        statuses[top].state.value in ("superseded", "deprecated")
    return {
        "id": q["id"],
        "category": q["category"],
        "hit@1": float(first_good == 0),
        "recall@5": len(expected & set(ranked[:5])) / len(expected),  # over the top 5 documents
        "mrr": 1.0 / (first_good + 1) if first_good is not None else 0.0,
        "context_recall": len(expected & context) / len(expected),
        "stale_first": float(stale_first),
        "distinct_docs_in_top8": len({h.passage.doc_id for h in retrieval.hits}),
        "missing_from_context": sorted(expected - context),
    }


def build_index(config: Config, docs, cache: dict) -> HandbookIndex:
    embeddings = None
    if config.model:
        key = (config.model, config.prefix)
        if key not in cache:
            embedder = FastEmbedder(config.model, query_prefix=config.prefix)
            cache[key] = EmbeddingCache(embedder, cache_path_for(embedder))
        embeddings = cache[key]
    return HandbookIndex.build(docs, mode=config.mode, embeddings=embeddings)


def run(config: Config, questions: list[dict], docs, cache: dict) -> list[dict]:
    index = build_index(config, docs, cache)
    rows = []
    for q in questions:
        ref = reference_date_for(q)
        start = time.perf_counter()
        history = q.get("history") or []
        previous = history[-1]["question"] if history and is_follow_up(q["question"]) else None
        retrieval = index.retrieve(q["question"], ref, use_validity=config.use_validity,
                                   follow_references=config.follow_references, follow_up_of=previous)
        row = score_question(q, retrieval, index.statuses(ref))
        row["latency_ms"] = (time.perf_counter() - start) * 1000
        row["context_docs"] = len(retrieval.context)
        row["context_chars"] = sum(e - s for c in retrieval.context for s, e in c.spans)
        rows.append(row)
    return rows


METRICS = ["hit@1", "recall@5", "mrr", "context_recall", "stale_first"]


def summarise(rows: list[dict]) -> dict:
    out = {m: statistics.mean(r[m] for r in rows) for m in METRICS}
    out["latency_ms_p50"] = statistics.median(r["latency_ms"] for r in rows)
    out["context_docs_mean"] = statistics.mean(r["context_docs"] for r in rows)
    out["context_chars_mean"] = statistics.mean(r["context_chars"] for r in rows)
    return out


def wins_losses(rows: list[dict], baseline: list[dict], metric: str) -> tuple[int, int, float]:
    """Questions where this configuration beats / loses to the baseline, with a two-sided sign test."""
    base = {r["id"]: r[metric] for r in baseline}
    wins = sum(r[metric] > base[r["id"]] for r in rows)
    losses = sum(r[metric] < base[r["id"]] for r in rows)
    n = wins + losses
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    return wins, losses, p


def by_category(rows: list[dict], metric: str) -> dict[str, float]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r[metric])
    return {c: statistics.mean(v) for c, v in sorted(groups.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="dev", help="question set: dev (default), holdout or followup")
    parser.add_argument("--configs", nargs="*", help="only these configuration names")
    args = parser.parse_args(argv)

    questions = [q for q in load_questions(args.set) if q["expected_doc_ids"]]  # refusals have no gold documents
    docs, _ = load_corpus(CORPUS_DIR)
    configs = [c for c in CONFIGS if not args.configs or c.name in args.configs or c.name == BASELINE]

    cache: dict = {}
    results = {c.name: {"config": c.__dict__, "questions": run(c, questions, docs, cache)} for c in configs}
    for r in results.values():
        r["summary"] = summarise(r["questions"])
        r["context_recall_by_category"] = by_category(r["questions"], "context_recall")
        base = results[BASELINE]["questions"]
        r["vs_bm25"] = {m: dict(zip(("wins", "losses", "sign_test_p"), wins_losses(r["questions"], base, m),
                                    strict=True)) for m in ("hit@1", "context_recall")}

    print(f"\nRetrieval on the {args.set} set: {len(questions)} questions with gold documents, 0 model calls\n")
    header = (f"{'configuration':32} {'hit@1':>6} {'rec@5':>6} {'MRR':>6} {'ctx rec':>8} {'stale1st':>8} "
              f"{'docs':>5} {'chars':>6} {'hit@1 vs bm25':>15}")
    print(header + "\n" + "-" * len(header))
    for name, r in results.items():
        s, v = r["summary"], r["vs_bm25"]["hit@1"]
        versus = "-" if name == BASELINE else f"+{v['wins']}/-{v['losses']} p={v['sign_test_p']:.2f}"
        print(f"{name:32} {s['hit@1']:6.3f} {s['recall@5']:6.3f} {s['mrr']:6.3f} {s['context_recall']:8.3f} "
              f"{s['stale_first']:8.3f} {s['context_docs_mean']:5.1f} {s['context_chars_mean']:6.0f} {versus:>15}")

    chosen = "hybrid bge-small"
    if chosen in results:
        print(f"\nContext recall by category ({chosen}):")
        for cat, value in results[chosen]["context_recall_by_category"].items():
            print(f"  {cat:14} {value:.3f}")
        misses = [(r["id"], r["missing_from_context"]) for r in results[chosen]["questions"] if r["missing_from_context"]]
        print(f"Context misses: {misses or 'none'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval-{args.set}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nPer-question results: {out.relative_to(ROOT)}")
    return results


if __name__ == "__main__":
    main()
