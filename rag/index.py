import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from rag.chunking import Passage, chunk_documents
from rag.config import reference_date
from rag.manifest import load_corpus
from rag.models import Document, MetadataFinding
from rag.retrieval.embeddings import EmbeddingCache, FastEmbedder, cache_path_for
from rag.retrieval.search import (
    ContextDoc,
    Hit,
    ReferenceFinder,
    Retriever,
    build_context,
)
from rag.retrieval.spelling import SpellFixer
from rag.retrieval.vector_store import ChromaStore, VectorStoreError, store_from_env
from rag.validity import DocStatus, ValidityFinding, assess

log = logging.getLogger(__name__)

# Chosen with evals/retrieval_eval.py on the dev set (results in evals/results/retrieval-dev.json):
# hybrid BM25 + bge-small, no query prefix (it changed no dev question once fused with BM25).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODE = "hybrid"
TOP_K = 8

# Questions whose answer depends on the date ("Can I deploy today?", "Is there a freeze on December 15?").
DATED_QUESTION = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|now|right now|currently|this (week|weekend|month|year|quarter)|"
    r"next (week|month|year)|upcoming|soon)\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}\b|\b20\d\d-\d\d-\d\d\b",
    re.IGNORECASE)
_ISO_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
_SCHEDULE_WORDS = {"calendar", "schedule", "schedules", "rota", "roster", "timetable", "dates", "plan", "list"}
SCHEDULE_HITS = 3  # like cross-references: only the top passages can bring a schedule in


def schedule_subjects(docs: list[Document]) -> dict[str, re.Pattern]:
    """Dated schedules (most table rows carry a date) and what they list, from the title:
    the Release Calendar lists releases. Used to answer "today" questions from rules like
    "a freeze starts 72 hours before every scheduled release", which never name the calendar."""
    out = {}
    for doc in docs:
        rows = [b for b in doc.blocks if b.kind == "table_row"]
        dated = sum(bool(_ISO_DATE.search(b.text)) for b in rows)
        words = [w for w in re.findall(r"[a-z]+", doc.meta.title.lower()) if w not in _SCHEDULE_WORDS and len(w) >= 4]
        if dated >= 3 and dated * 2 >= len(rows) and words:
            out[doc.doc_id] = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")s?\b", re.IGNORECASE)
    return out


def load_embeddings(model_name: str | None = None) -> EmbeddingCache:
    name = model_name or os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    embedder = FastEmbedder(name)
    return EmbeddingCache(embedder, cache_path_for(embedder))


@dataclass(frozen=True)
class Retrieval:
    """What the answer step gets for one question."""

    question: str
    reference_date: date
    hits: list[Hit]
    context: list[ContextDoc]
    pool: list[Hit]  # the full ranking, for metrics and debugging
    top_bm25: float | None
    top_cosine: float | None
    corrections: list[tuple[str, str]] = field(default_factory=list)  # typos also searched as (typed, meant)


@dataclass
class HandbookIndex:
    """Documents, passages and the retriever. Document states are worked out per request date."""

    docs: dict[str, Document]
    passages: list[Passage]
    retriever: Retriever
    default_date: date | None = None  # None: today in the company time zone, on every request
    findings: list[MetadataFinding] = field(default_factory=list)

    def __post_init__(self):
        # links between documents (cross-references, schedules) only ever lead to official ones
        official = [d for d in self.docs.values() if not d.meta.uploaded]
        self.references = ReferenceFinder(official)
        self.passages_by_doc: dict[str, list[Passage]] = defaultdict(list)
        for p in self.passages:
            self.passages_by_doc[p.doc_id].append(p)
        self._statuses: dict[date, tuple[dict[str, DocStatus], list[ValidityFinding]]] = {}
        self.schedules = schedule_subjects(official)
        self.spelling = SpellFixer(d.text for d in self.docs.values())  # uploads' words count too

    @property
    def mode(self) -> str:
        return self.retriever.mode

    def date_for(self, ref: date | str | None = None) -> date:
        return reference_date(ref if ref is not None else self.default_date)

    def statuses(self, ref: date | str | None = None) -> dict[str, DocStatus]:
        """Document states on `ref` (default: the index's date, else today). Cached per date."""
        day = self.date_for(ref)
        if day not in self._statuses:
            if len(self._statuses) > 32:  # a long-running server sees one new date a day
                self._statuses.clear()
            self._statuses[day] = assess([d.meta for d in self.docs.values()], day)
        return self._statuses[day][0]

    def validity_findings(self, ref: date | str | None = None) -> list[ValidityFinding]:
        self.statuses(ref)
        return self._statuses[self.date_for(ref)][1]

    @classmethod
    def build(cls, docs: list[Document], ref: date | str | None = None, mode: str | None = None,
              embeddings: EmbeddingCache | None = None, findings: list | None = None,
              vector_store: ChromaStore | None = None) -> "HandbookIndex":
        """Build from loaded documents. Falls back to BM25 if the embedding model cannot load.

        Vectors are stored in and searched through ChromaDB (CHROMA_URL), unless VECTOR_STORE=memory;
        an unreachable Chroma falls back to the in-memory vectors (see rag.retrieval.vector_store).
        """
        mode = mode or os.environ.get("RETRIEVAL_MODE", "").strip() or DEFAULT_MODE
        if mode != "bm25" and embeddings is None:
            try:
                embeddings = load_embeddings()
            except Exception as exc:  # offline, missing model files, broken install
                log.warning("embedding model unavailable (%s); falling back to keyword search", exc)
                mode = "bm25"
        if mode != "bm25" and vector_store is None:
            try:
                vector_store = store_from_env(embeddings.embedder.name)
            except VectorStoreError as exc:
                log.error("%s; searching the in-memory vectors instead", exc)
        passages = chunk_documents(docs)
        retriever = Retriever(docs, passages, embeddings, mode=mode, vector_store=vector_store)
        default = reference_date(ref) if ref is not None else None
        return cls({d.doc_id: d for d in docs}, passages, retriever, default, list(findings or []))

    @classmethod
    def from_corpus(cls, corpus_dir: Path, ref: date | str | None = None, mode: str | None = None,
                    embeddings: EmbeddingCache | None = None) -> "HandbookIndex":
        docs, findings = load_corpus(corpus_dir)
        return cls.build(docs, ref, mode, embeddings, findings)

    def retrieve(self, question: str, ref: date | str | None = None, k: int = TOP_K,
                 use_validity: bool = True, follow_references: bool = True,
                 follow_up_of: str | None = None) -> Retrieval:
        """Search for `question`. For a follow-up ("which one lives in Austin?"), pass the previous
        question as `follow_up_of`: results for the question alone and for both together are merged,
        so the follow-up finds what it refers to while a new topic is not swamped by the old one."""
        day = self.date_for(ref)
        statuses = self.statuses(day)
        states = statuses if use_validity else None
        corrections = self.spelling.corrections(question)
        query = self.spelling.expand(question)  # "anual leves" is also searched as "annual leave"
        result = self.retriever.search(query, k=k, statuses=states)
        hits = result.hits
        if follow_up_of and follow_up_of.strip():
            combined = self.retriever.search(f"{self.spelling.expand(follow_up_of.strip())} {query}", k=k,
                                             statuses=states)
            hits = _interleave(result.hits, combined.hits, k)
        schedules = []
        if follow_references and DATED_QUESTION.search(question):
            top = " ".join(h.passage.text for h in hits[:SCHEDULE_HITS])
            schedules = [d for d, subject in self.schedules.items() if subject.search(top)]
        context = build_context(hits, self.docs, statuses, result.pool, self.passages_by_doc,
                                self.references if follow_references else None, schedules=schedules)
        return Retrieval(question, day, hits, context, result.pool, result.top_bm25, result.top_cosine, corrections)


def _interleave(first: list[Hit], second: list[Hit], k: int, max_per_doc: int = 2) -> list[Hit]:
    """Alternate two rankings (first, second, first, ...), skipping repeats, at most `max_per_doc` per document."""
    merged, seen, per_doc = [], set(), {}
    for pair in zip(first, second, strict=False):
        for hit in pair:
            pid, doc = hit.passage.passage_id, hit.passage.doc_id
            if pid in seen or per_doc.get(doc, 0) >= max_per_doc:
                continue
            seen.add(pid)
            per_doc[doc] = per_doc.get(doc, 0) + 1
            merged.append(hit)
    for hit in first[len(second):] + second[len(first):]:  # leftovers when one list is longer
        pid, doc = hit.passage.passage_id, hit.passage.doc_id
        if pid not in seen and per_doc.get(doc, 0) < max_per_doc:
            seen.add(pid)
            per_doc[doc] = per_doc.get(doc, 0) + 1
            merged.append(hit)
    return merged[:k]
