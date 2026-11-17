import re
from dataclasses import dataclass, field

import numpy as np

from rag.chunking import Passage
from rag.models import Document
from rag.retrieval.bm25 import BM25
from rag.retrieval.embeddings import EmbeddingCache
from rag.retrieval.text import tokenize
from rag.validity import DocStatus, State, describe

RRF_K = 60  # standard reciprocal-rank-fusion constant; ranks matter, raw scores do not

# Validity as a rank penalty: superseded and deprecated passages drop this many places
# in each ranking before fusion. Fused scores sit in a narrow band (top 2/61), so a
# multiplier like x0.9 would silently cost ~8 places; a stated number of places is easy
# to explain and check. Stale versions stay findable ("what did v1 say?") but fall well
# below the version in force. Meeting notes and upcoming revisions are not penalised:
# they are the only source for some answers, and the answer step sees every state.
# (Measured on dev: penalising them by 2 places cost 2 points of hit@1 for nothing.)
STALE_RANK_PENALTY = 30
REFERENCE_HITS = 3  # cross-references are followed from the top passages only (same recall, -24% text)

MAX_PASSAGES_PER_DOC = 2  # small documents are sent whole anyway; leave room for other documents
SMALL_DOC_CHARS = 2500  # documents up to this size are sent whole (all but the on-call runbook)
CONTEXT_BUDGET_CHARS = 12000  # about 3k tokens of handbook text per question


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    rank: int


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]  # top k, at most MAX_PASSAGES_PER_DOC per document
    pool: list[Hit]  # the full ranking, for metrics and for picking related passages
    # Raw relevance hints for the answer step. They overlap between answerable and
    # unanswerable questions, so they are not used as a cut-off here.
    top_bm25: float | None
    top_cosine: float | None


@dataclass
class ContextDoc:
    """One document's share of the answer context."""

    doc_id: str
    title: str
    state: State
    status_note: str  # e.g. "Superseded by Leave Policy v2."
    reason: str  # "retrieved", or why it was added ("current version of ...")
    spans: list[tuple[int, int]] = field(default_factory=list)  # merged (start, end) offsets
    whole_document: bool = False

    def text(self, doc: Document) -> str:
        return "\n[...]\n".join(doc.text[s:e] for s, e in self.spans)


def _ranks(scores: np.ndarray) -> list[int]:
    """Passage indices, best first, keeping only those with a positive score."""
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order if scores[i] > 0]


class Retriever:
    """Rank passages with BM25, embeddings, or both (reciprocal rank fusion).

    Document states are passed per search, so a long-running server ranks with
    today's states (the 2027 revision takes effect without a restart).
    """

    def __init__(self, docs: list[Document], passages: list[Passage],
                 embeddings: EmbeddingCache | None = None, mode: str = "hybrid"):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if mode != "bm25" and embeddings is None:
            raise ValueError(f"mode {mode!r} needs an embedding model")
        self.titles = {d.doc_id: d.meta.title for d in docs}
        self.passages = passages
        self.mode = mode
        self.bm25 = BM25([p.search_text for p in passages]) if mode != "dense" else None
        self.embeddings = embeddings
        self.matrix = embeddings.get([p.search_text for p in passages]) if mode != "bm25" else None

    def _penalties(self, statuses: dict[str, DocStatus], question: str) -> np.ndarray:
        asked = set(tokenize(question))
        by_doc: dict[str, int] = {}
        for doc_id, status in statuses.items():
            if status.state in (State.SUPERSEDED, State.DEPRECATED):
                # a question naming what sets this old version apart ("v1", "2023") asks for it on purpose
                current = self.titles.get(status.current_version or "", "")
                markers = set(tokenize(self.titles.get(doc_id, ""))) - set(tokenize(current))
                by_doc[doc_id] = 0 if markers & asked else STALE_RANK_PENALTY
        return np.array([by_doc.get(p.doc_id, 0) for p in self.passages], dtype=float)

    def search(self, question: str, k: int = 8, statuses: dict[str, DocStatus] | None = None,
               max_per_doc: int = MAX_PASSAGES_PER_DOC) -> SearchResult:
        """Top-k passages; pass `statuses` for validity-aware ranking, or None to rank on text alone."""
        if not self.passages or not question.strip():
            return SearchResult([], [], None, None)
        rankings: list[list[int]] = []
        top_bm25 = top_cosine = None
        if self.bm25 is not None:
            scores = np.array(self.bm25.scores(question))
            top_bm25 = float(scores.max())
            rankings.append(_ranks(scores))
        if self.matrix is not None:
            cosine = self.matrix @ self.embeddings.embedder.embed_query(question)
            top_cosine = float(cosine.max())
            rankings.append(_ranks(cosine + 1.0))  # shifted to (0, 2] so every passage ranks

        penalty = self._penalties(statuses, question) if statuses else np.zeros(len(self.passages))
        fused = np.zeros(len(self.passages))
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] += 1.0 / (RRF_K + rank + 1 + penalty[i])

        pool = [Hit(self.passages[i], float(fused[i]), r + 1) for r, i in enumerate(_ranks(fused))]
        hits, per_doc = [], {}
        for hit in pool:
            doc_id = hit.passage.doc_id
            if per_doc.get(doc_id, 0) < max_per_doc:
                per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
                hits.append(hit)
                if len(hits) == k:
                    break
        return SearchResult(hits, pool, top_bm25, top_cosine)


_GENERIC_TITLE_WORDS = {"policy", "guide", "guidelines", "process", "procedure", "runbook", "rules", "overview"}


class ReferenceFinder:
    """Find the documents a piece of text names ("see the VPN Guide", "after the introductory period").

    A title matches in full, or without a trailing generic word when at least two
    words remain ("introductory period" for the Introductory Period Policy), so a
    one-word remainder like "travel" never links on its own.
    """

    def __init__(self, docs: list[Document]):
        self.patterns = {}
        for doc in docs:
            title = doc.meta.title
            phrases = {title}
            words = title.split()
            if len(words) >= 3 and words[-1].lower() in _GENERIC_TITLE_WORDS:
                phrases.add(" ".join(words[:-1]))
            alternatives = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
            self.patterns[doc.doc_id] = re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])", re.IGNORECASE)

    def find(self, text: str, exclude: str | None = None) -> list[str]:
        """Named documents, in the order they first appear in the text."""
        found = []
        for doc_id, pattern in self.patterns.items():
            if doc_id != exclude and (m := pattern.search(text)):
                found.append((m.start(), doc_id))
        return [doc_id for _, doc_id in sorted(found)]


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching spans (a table row inside its whole-table passage, etc.)."""
    merged: list[list[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def build_context(hits: list[Hit], docs: dict[str, Document], statuses: dict[str, DocStatus],
                  pool: list[Hit] | None = None, passages_by_doc: dict[str, list[Passage]] | None = None,
                  references: ReferenceFinder | None = None, reference_hits: int = REFERENCE_HITS,
                  small_doc_chars: int = SMALL_DOC_CHARS, budget: int = CONTEXT_BUDGET_CHARS) -> list[ContextDoc]:
    """Turn ranked passages into the documents and spans the answer step sees.

    - small documents are sent whole, so facts split across sections (city tier in
      section 3, hotel rate in 4.2) arrive together
    - overlapping passages from one document are merged into one slot
    - a stale document brings its current version; a document in force brings its
      upcoming revision and amendments; each says why it was added
    - documents named in one of the top `reference_hits` passages ("after the
      introductory period") are added after all retrieved ones, if in force and within budget
    - a related large document contributes its best-ranked passage for this question
    - stops once the text budget is used, always keeping the top-ranked document
    """
    titles = {d: doc.meta.title for d, doc in docs.items()}
    rank_of = {h.passage.passage_id: h.rank for h in (pool or hits)}
    out: dict[str, ContextDoc] = {}
    used = 0

    def add(doc_id: str, span: tuple[int, int] | None, reason: str) -> bool:
        nonlocal used
        doc, status = docs.get(doc_id), statuses.get(doc_id)
        if doc is None:
            return False
        whole = len(doc.text) <= small_doc_chars or (span is None and len(doc.text) <= budget)
        new_span = (0, len(doc.text)) if whole or span is None else span
        entry = out.get(doc_id)
        before = sum(e - s for s, e in entry.spans) if entry else 0
        spans = _merge((entry.spans if entry else []) + [new_span])
        cost = sum(e - s for s, e in spans) - before
        if out and used + cost > budget:
            return False
        used += cost
        if entry is None:
            out[doc_id] = ContextDoc(doc_id, doc.meta.title, status.state if status else State.IN_FORCE,
                                     describe(status, titles) if status else "In force.", reason)
            entry = out[doc_id]
        entry.spans = spans
        entry.whole_document = entry.whole_document or (spans == [(0, len(doc.text))])
        return True

    def best_span(doc_id: str) -> tuple[int, int] | None:
        """A related document's best-ranked passage for this question, else its first passage."""
        candidates = (passages_by_doc or {}).get(doc_id) or []
        if not candidates:
            return None
        best = min(candidates, key=lambda p: rank_of.get(p.passage_id, float("inf")))
        return best.start, best.end

    retrieved_text: list[tuple[str, str]] = []
    for n, hit in enumerate(hits):
        p = hit.passage
        if not add(p.doc_id, (p.start, p.end), "retrieved"):
            continue
        if n < reference_hits:
            retrieved_text.append((p.doc_id, p.text))
        status = statuses.get(p.doc_id)
        if status is None:
            continue
        name = titles.get(p.doc_id, p.doc_id)
        related = [(status.current_version, f"current version of {name}")] if status.current_version else []
        related += [(d, f"upcoming revision of {name}") for d in status.upcoming]
        related += [(d, f"amends {name}") for d in status.amended_by]
        for doc_id, reason in related:
            if doc_id not in out:
                add(doc_id, best_span(doc_id), reason)

    if references is not None:
        for source, text in retrieved_text:
            for doc_id in references.find(text, exclude=source):
                status = statuses.get(doc_id)
                if doc_id not in out and (status is None or status.in_force):
                    add(doc_id, best_span(doc_id), f"referenced by {titles.get(source, source)}")
    return list(out.values())
