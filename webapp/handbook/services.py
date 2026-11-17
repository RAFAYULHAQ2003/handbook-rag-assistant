"""Shared, lazily built objects for the web app: one search index and one model client per process.

Building the index loads 64 documents and the embedding model (a few seconds), so it is done
once, on first use, behind a lock: parallel requests during start-up wait for the same build
instead of each starting their own.

Uploaded documents live in their own folder. When it changes (an upload here or in another worker
process), the next request rebuilds the index from the already-parsed handbook plus the uploads,
reusing the loaded embedding model and the embedding cache, so only new passages are embedded.
"""

import logging
import threading
from datetime import date

from django.conf import settings

from rag.answer import STATE_LABEL, Answer
from rag.index import HandbookIndex
from rag.llm import LLMClient, LLMError, client_from_env
from rag.retrieval.search import ContextDoc
from rag.uploads import UploadStore

log = logging.getLogger(__name__)

_lock = threading.Lock()
_index: HandbookIndex | None = None
_index_uploads: int | None = None  # the uploads version the index was built with
_client: LLMClient | None = None
_client_error: str | None = None


def upload_store() -> UploadStore:
    return UploadStore(settings.UPLOAD_DIR, settings.UPLOAD_MAX_BYTES, settings.UPLOAD_MAX_DOCUMENTS)


def get_index() -> HandbookIndex:
    global _index, _index_uploads
    store = upload_store()
    version = store.version()
    if _index is None or version != _index_uploads:
        with _lock:
            version = store.version()
            if _index is None:
                _index = HandbookIndex.from_corpus(settings.CORPUS_DIR)
                _index_uploads = 0
                log.info("handbook index ready: %d documents, %d passages, mode %s",
                         len(_index.docs), len(_index.passages), _index.mode)
            if version != _index_uploads:
                official = [d for d in _index.docs.values() if not d.meta.uploaded]
                uploads = store.load()
                _index = HandbookIndex.build(official + uploads, _index.default_date, _index.mode,
                                             _index.retriever.embeddings, _index.findings)
                _index_uploads = version
                log.info("index rebuilt with %d uploaded document(s)", len(uploads))
    return _index


def is_uploaded(doc_id: str) -> bool:
    doc = _index.docs.get(doc_id) if _index is not None else None
    return bool(doc and doc.meta.uploaded)


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
    """Forget the cached index and client (used by tests and after uploads)."""
    global _index, _index_uploads, _client, _client_error
    with _lock:
        _index = _index_uploads = _client = _client_error = None


# ---------- turning results into JSON-friendly dicts ----------

def context_json(context: list[ContextDoc]) -> list[dict]:
    return [{"doc_id": c.doc_id, "title": c.title, "state": c.state.value, "state_label": STATE_LABEL[c.state],
             "status_note": c.status_note, "reason": c.reason, "uploaded": is_uploaded(c.doc_id)} for c in context]


def answer_json(a: Answer, source_url) -> dict:
    """The API's answer shape. `source_url(doc_id, start, end)` builds the link that opens a citation."""
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
             "url": source_url(c.doc_id, c.start, c.end), "uploaded": is_uploaded(c.doc_id)}
            for i, c in enumerate(a.citations, 1)
        ],
        "dropped_citations": a.dropped_citations,
        "reference_date": a.retrieval.reference_date.isoformat() if a.retrieval else None,
        "search_corrections": [{"typed": t, "searched": f} for t, f in a.retrieval.corrections] if a.retrieval else [],
        "context": context_json(a.retrieval.context) if a.retrieval else [],
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
