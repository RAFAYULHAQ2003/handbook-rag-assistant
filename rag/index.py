import logging
import os
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
from rag.validity import DocStatus, ValidityFinding, assess

log = logging.getLogger(__name__)

# Chosen with evals/retrieval_eval.py on the dev set (results in evals/results/retrieval-dev.json):
# hybrid BM25 + bge-small, no query prefix (it changed no dev question once fused with BM25).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MODE = "hybrid"
TOP_K = 8


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


@dataclass
class HandbookIndex:
    """Documents, passages and the retriever. Document states are worked out per request date."""

    docs: dict[str, Document]
    passages: list[Passage]
    retriever: Retriever
    default_date: date | None = None  # None: today in the company time zone, on every request
    findings: list[MetadataFinding] = field(default_factory=list)

    def __post_init__(self):
        self.references = ReferenceFinder(list(self.docs.values()))
        self.passages_by_doc: dict[str, list[Passage]] = defaultdict(list)
        for p in self.passages:
            self.passages_by_doc[p.doc_id].append(p)
        self._statuses: dict[date, tuple[dict[str, DocStatus], list[ValidityFinding]]] = {}

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
              embeddings: EmbeddingCache | None = None, findings: list | None = None) -> "HandbookIndex":
        """Build from loaded documents. Falls back to BM25 if the embedding model cannot load."""
        mode = mode or os.environ.get("RETRIEVAL_MODE", "").strip() or DEFAULT_MODE
        if mode != "bm25" and embeddings is None:
            try:
                embeddings = load_embeddings()
            except Exception as exc:  # offline, missing model files, broken install
                log.warning("embedding model unavailable (%s); falling back to keyword search", exc)
                mode = "bm25"
        passages = chunk_documents(docs)
        retriever = Retriever(docs, passages, embeddings, mode=mode)
        default = reference_date(ref) if ref is not None else None
        return cls({d.doc_id: d for d in docs}, passages, retriever, default, list(findings or []))

    @classmethod
    def from_corpus(cls, corpus_dir: Path, ref: date | str | None = None, mode: str | None = None,
                    embeddings: EmbeddingCache | None = None) -> "HandbookIndex":
        docs, findings = load_corpus(corpus_dir)
        return cls.build(docs, ref, mode, embeddings, findings)

    def retrieve(self, question: str, ref: date | str | None = None, k: int = TOP_K,
                 use_validity: bool = True, follow_references: bool = True) -> Retrieval:
        day = self.date_for(ref)
        statuses = self.statuses(day)
        result = self.retriever.search(question, k=k, statuses=statuses if use_validity else None)
        context = build_context(result.hits, self.docs, statuses, result.pool, self.passages_by_doc,
                                self.references if follow_references else None)
        return Retrieval(question, day, result.hits, context, result.pool, result.top_bm25, result.top_cosine)
