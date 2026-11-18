"""Shared, lazily built objects for the web app: one search index, one model client and one store of
conversation uploads per process.

Building the index loads 64 documents and the embedding model (a few seconds), so it is done
once, on first use, behind a lock: parallel requests during start-up wait for the same build
instead of each starting their own. Uploads never change it: they belong to one conversation
(rag.uploads.SessionUploads) and are searched alongside it.
"""

import logging
import os
import threading
from datetime import date

from django.conf import settings

from rag.answer import STATE_LABEL, Answer
from rag.index import HandbookIndex
from rag.llm import LLMClient, LLMError, client_from_env
from rag.retrieval.search import ContextDoc
from rag.retrieval.vector_store import ChromaStore
from rag.uploads import SessionUploads

log = logging.getLogger(__name__)

_lock = threading.Lock()
_index: HandbookIndex | None = None
_uploads: SessionUploads | None = None
_client: LLMClient | None = None
_client_error: str | None = None


def get_index() -> HandbookIndex:
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = HandbookIndex.from_corpus(settings.CORPUS_DIR)
                log.info("handbook index ready: %d documents, %d passages, mode %s, vectors in %s",
                         len(_index.docs), len(_index.passages), _index.mode, _index.retriever.vector_store_name)
    return _index


def uploads() -> SessionUploads:
    """Each conversation's uploads: chunked and embedded with the handbook's model, kept apart, temporary."""
    global _uploads
    if _uploads is None:
        index = get_index()
        embeddings = index.retriever.embeddings
        embedder = embeddings.embedder if embeddings is not None else None
        store_factory = None
        if isinstance(index.retriever.store, ChromaStore):  # the handbook's vectors are in Chroma: so are these
            def store_factory(sid, model=embedder.name):
                return _session_store(sid, model)
        with _lock:
            if _uploads is None:
                _uploads = SessionUploads(settings.UPLOAD_DIR, embedder, index.mode, store_factory,
                                          ttl_seconds=settings.UPLOAD_SESSION_TTL_MINUTES * 60,
                                          max_bytes=settings.UPLOAD_MAX_BYTES,
                                          max_documents=settings.UPLOAD_MAX_DOCUMENTS)
    return _uploads


def _session_store(sid: str, model_name: str) -> ChromaStore:
    """A temporary Chroma collection for one conversation: "conversation-<id>-<model>"."""
    url = os.environ.get("CHROMA_URL", "").strip() or "http://localhost:8001"
    return ChromaStore.connect(url, model_name, prefix=f"conversation-{sid}")


def get_client() -> LLMClient | None:
    """The model client, or None if it is not configured (the reason is in client_error())."""
    global _client, _client_error
    if _client is None and _client_error is None:
        with _lock:
            if _client is None and _client_error is None:
                try:
                    _client = client_from_env()
                except LLMError as exc:
                    _client_error = str(exc)
                    log.error("model client not configured: %s", exc)
    return _client


def client_error() -> str | None:
    get_client()
    return _client_error


def reset() -> None:
    """Forget the cached index, uploads store and client (used by tests)."""
    global _index, _uploads, _client, _client_error
    with _lock:
        _index = _uploads = _client = _client_error = None


# ---------- turning results into JSON-friendly dicts ----------

def context_json(context: list[ContextDoc], uploaded: set[str] = frozenset()) -> list[dict]:
    return [{"doc_id": c.doc_id, "title": c.title, "state": c.state.value, "state_label": STATE_LABEL[c.state],
             "status_note": c.status_note, "reason": c.reason, "uploaded": c.doc_id in uploaded} for c in context]


def answer_json(a: Answer, source_url) -> dict:
    """The API's answer shape. `source_url(doc_id, start, end)` builds the link that opens a citation."""
    uploaded = set(a.retrieval.docs) if a.retrieval else set()
    return {
        "question": a.question,
        "answer": a.answer,
        "verdict": a.verdict,
        "answerable": a.answerable,
        "needs_clarification": a.needs_clarification,
        "unsupported": a.unsupported,
        "citations": [
            {"n": i, "doc_id": c.doc_id, "title": c.title, "state": c.state.value,
             "state_label": STATE_LABEL[c.state], "quote": c.quote, "start": c.start, "end": c.end,
             "url": source_url(c.doc_id, c.start, c.end), "uploaded": c.doc_id in uploaded}
            for i, c in enumerate(a.citations, 1)
        ],
        "dropped_citations": a.dropped_citations,
        "reference_date": a.retrieval.reference_date.isoformat() if a.retrieval else None,
        "search_corrections": [{"typed": t, "searched": f} for t, f in a.retrieval.corrections] if a.retrieval else [],
        "context": context_json(a.retrieval.context, uploaded) if a.retrieval else [],
        "model": a.model,
        "cached": a.cached,
        "used_fallback": a.used_fallback,
        "small_talk": a.small_talk,
        "latency_ms": round(a.latency_ms),
        "error": a.error,
    }


def parse_as_of(value: str | None) -> date | None:
    """A user-supplied 'answer as of' date, or None for today. Raises ValueError if malformed."""
    if not value:
        return None
    day = date.fromisoformat(value.strip())
    if not date(2000, 1, 1) <= day <= date(2100, 12, 31):
        raise ValueError("date out of range")
    return day
