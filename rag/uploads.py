"""Documents uploaded in one conversation: parsed, chunked and embedded for that conversation only.

Each conversation has a random id (made by the page, sent with every request). Its uploads live in
a temporary folder and, for meaning search, a temporary ChromaDB collection of their own; they are
never added to the handbook index, to the shared embedding cache, or to anyone else's search.
Everything of a conversation (files, chunks, vectors) is deleted when the person starts a new
conversation, after `ttl_seconds` without use, or when the server restarts (the start script empties
the folder).

Uploads are never official: they are marked `uploaded`, so they cannot replace or amend a handbook
document (rag.validity), and the answer step is told they are not policy.

Several web worker processes can serve one conversation: the folder is the shared record, and a
worker that has not seen the conversation yet loads it from there.
"""

import hashlib
import io
import json
import logging
import re
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from rag.chunking import Passage, chunk_documents
from rag.loaders import LOADERS, EmptyDocument
from rag.manifest import load_document, slugify
from rag.models import Document
from rag.retrieval.embeddings import EmbeddingCache
from rag.retrieval.search import Retriever

log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = sorted(LOADERS)
MAX_UNZIPPED_BYTES = 50_000_000  # .docx, .xlsx and .pptx are zip files: refuse "zip bombs" before parsing
SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
MANIFEST = "uploads.json"
LAST_USED = ".last-used"


class UploadError(ValueError):
    """A problem with the uploaded file, worded for the person who uploaded it."""


def valid_session_id(value) -> str | None:
    value = str(value or "").strip().lower()
    return value if SESSION_ID.match(value) else None


def _clean_title(value: str) -> str:
    """A title safe to show and to put in the prompt: no quotes, angle brackets or control characters."""
    return re.sub(r"\s+", " ", re.sub(r'[\x00-\x1f"<>`]', " ", value)).strip()[:120] or "Uploaded document"


def _check_zip(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED_BYTES:
                raise UploadError("This file expands to more than 50 MB and cannot be read.")
    except zipfile.BadZipFile as exc:
        raise UploadError("This does not look like a valid Office file (.docx, .xlsx or .pptx).") from exc


def _mark(doc: Document) -> Document:
    doc.meta.uploaded = True
    doc.meta.supersedes = None  # an upload can never replace a handbook document
    doc.meta.in_manifest = False
    return doc


def parse_upload(filename: str, data: bytes, folder: Path, today: date, max_bytes: int) -> tuple[Document, dict]:
    """Check, store (in `folder`) and parse one file. Returns the document and its manifest entry."""
    ext = Path(filename or "").suffix.lower()
    if ext not in LOADERS:
        raise UploadError(f"Unsupported file type. Upload one of: {', '.join(ALLOWED_EXTENSIONS)}.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > max_bytes:
        raise UploadError(f"The file is larger than {max_bytes // 1_000_000} MB.")
    if ext in (".docx", ".xlsx", ".pptx"):
        _check_zip(data)
    digest = hashlib.sha256(data).hexdigest()
    doc_id = f"upload-{slugify(Path(filename).stem)[:40].strip('-')}-{digest[:8]}"
    stored = folder / f"{doc_id}{ext}"
    stored.write_bytes(data)
    try:
        parsed, _ = load_document(stored, {"doc_id": doc_id})
    except EmptyDocument as exc:
        stored.unlink(missing_ok=True)
        raise UploadError("No text could be found in this file. A scanned PDF is only an image of text "
                          "and needs OCR first; upload a PDF with selectable text instead.") from exc
    except Exception as exc:  # loaders wrap their errors, but a malformed file must never crash the page
        stored.unlink(missing_ok=True)
        raise UploadError(f"The file could not be read ({exc}).") from exc
    meta = parsed.meta
    title = meta.title if meta.title and meta.title != stored.stem else Path(filename).stem
    entry = {"path": stored.name, "doc_id": doc_id, "title": _clean_title(title), "category": "Uploaded",
             "owner": "Uploaded in this conversation", "effective_date": (meta.effective_date or today).isoformat(),
             "status": "current", "original_name": _clean_title(Path(filename).name)}
    doc, _ = load_document(stored, entry)
    return _mark(doc), entry


@dataclass
class Conversation:
    """One conversation's uploads, ready to search."""

    sid: str
    folder: Path
    docs: dict[str, Document] = field(default_factory=dict)
    passages_by_doc: dict[str, list[Passage]] = field(default_factory=dict)
    retriever: Retriever | None = None
    store: object | None = None  # its temporary ChromaStore, if any
    version: int = 0  # changes when the documents change


class SessionUploads:
    """Uploads per conversation, kept in `root/<conversation id>/` and deleted when it ends."""

    def __init__(self, root: Path, embedder=None, mode: str = "bm25", store_factory=None,
                 ttl_seconds: int = 3600, max_bytes: int = 5_000_000, max_documents: int = 10):
        self.root = Path(root)
        self.embedder = embedder  # the handbook's embedding model; these vectors stay in memory/Chroma only
        self.mode = mode if embedder is not None else "bm25"
        self.store_factory = store_factory  # sid -> ChromaStore for that conversation, or None
        self.ttl = ttl_seconds
        self.max_bytes = max_bytes
        self.max_documents = max_documents
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.RLock()

    # ---------- reading ----------

    def get(self, sid: str | None) -> Conversation | None:
        """The conversation's uploads (loaded from disk if another worker made them), or None."""
        sid = valid_session_id(sid)
        if not sid:
            return None
        self.sweep()
        folder = self.root / sid
        manifest = folder / MANIFEST
        if not manifest.exists():
            with self._lock:
                self._conversations.pop(sid, None)
            return None
        version = manifest.stat().st_mtime_ns
        with self._lock:
            conv = self._conversations.get(sid)
            if conv is None or conv.version != version:
                conv = self._load(sid, folder, version, previous=conv)
                self._conversations[sid] = conv
            (folder / LAST_USED).touch()
            return conv

    def documents(self, sid: str | None) -> list[Document]:
        conv = self.get(sid)
        return list(conv.docs.values()) if conv else []

    # ---------- changing ----------

    def add(self, sid: str, filename: str, data: bytes, today: date) -> tuple[Document, bool]:
        """Parse, chunk and embed one file for this conversation. Returns (document, created)."""
        sid = valid_session_id(sid)
        if not sid:
            raise UploadError("Missing conversation id; reload the page and try again.")
        folder = self.root / sid
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            entries = self._entries(folder)
            digest = hashlib.sha256(data).hexdigest()[:8] if data else ""
            existing = next((e for e in entries if digest and e["doc_id"].endswith(f"-{digest}")), None)
            if existing:
                return self.get(sid).docs[existing["doc_id"]], False
            if len(entries) >= self.max_documents:
                raise UploadError(f"This conversation already has {self.max_documents} documents. "
                                  "Remove one, or start a new conversation.")
            with tempfile.TemporaryDirectory(dir=folder, prefix=".incoming-") as tmp:
                doc, entry = parse_upload(filename, data, Path(tmp), today, self.max_bytes)
                (Path(tmp) / entry["path"]).replace(folder / entry["path"])
            self._write(folder, entries + [entry])
        return self.get(sid).docs[doc.doc_id], True

    def remove(self, sid: str | None, doc_id: str) -> bool:
        sid = valid_session_id(sid)
        folder = self.root / sid if sid else None
        if folder is None or not folder.exists():
            return False
        with self._lock:
            entries = self._entries(folder)
            keep = [e for e in entries if e["doc_id"] != doc_id]
            if len(keep) == len(entries):
                return False
            for e in entries:
                if e["doc_id"] == doc_id:
                    (folder / Path(e["path"]).name).unlink(missing_ok=True)
            self._write(folder, keep)
        return True

    def clear(self, sid: str | None) -> None:
        """Delete everything of a conversation: files, chunks and vectors."""
        sid = valid_session_id(sid)
        if not sid:
            return
        with self._lock:
            conv = self._conversations.pop(sid, None)
            self._drop_vectors(sid, conv)
            shutil.rmtree(self.root / sid, ignore_errors=True)

    def sweep(self, now: float | None = None) -> int:
        """Delete conversations idle for longer than the time limit. Returns how many were removed."""
        now = now or time.time()
        if not self.root.exists():
            return 0
        removed = 0
        for folder in self.root.iterdir():
            if not folder.is_dir() or not SESSION_ID.match(folder.name):
                continue
            marker = folder / LAST_USED
            last = marker.stat().st_mtime if marker.exists() else folder.stat().st_mtime
            if now - last > self.ttl:
                self.clear(folder.name)
                removed += 1
        if removed:
            log.info("removed %d idle conversation(s) with uploads", removed)
        return removed

    # ---------- internals ----------

    def _entries(self, folder: Path) -> list[dict]:
        try:
            data = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return [e for e in data.get("documents", []) if isinstance(e, dict) and e.get("doc_id") and e.get("path")]

    def _write(self, folder: Path, entries: list[dict]) -> None:
        tmp = folder / f".{MANIFEST}.tmp"
        tmp.write_text(json.dumps({"documents": entries}, indent=2), encoding="utf-8")
        tmp.replace(folder / MANIFEST)

    def _load(self, sid: str, folder: Path, version: int, previous: Conversation | None) -> Conversation:
        docs = []
        for entry in self._entries(folder):
            try:
                doc, _ = load_document(folder / Path(entry["path"]).name, entry)
                docs.append(_mark(doc))
            except Exception as exc:  # a file that no longer parses must not break the conversation
                log.warning("skipping upload %s: %s", entry["doc_id"], exc)
        conv = Conversation(sid, folder, {d.doc_id: d for d in docs}, version=version)
        conv.store = previous.store if previous else None
        if not docs:
            self._drop_vectors(sid, conv)
            conv.store = None
            return conv
        passages = chunk_documents(docs)
        for p in passages:
            conv.passages_by_doc.setdefault(p.doc_id, []).append(p)
        # vectors made for this conversation only: an in-memory cache, never the shared cache on disk
        embeddings = EmbeddingCache(self.embedder, None) if self.embedder is not None else None
        if conv.store is None and self.store_factory is not None and embeddings is not None:
            try:
                conv.store = self.store_factory(sid)
            except Exception as exc:
                log.warning("temporary vector store unavailable (%s); searching in memory", exc)
        conv.retriever = Retriever(docs, passages, embeddings, mode=self.mode, vector_store=conv.store)
        return conv

    def _drop_vectors(self, sid: str, conv: Conversation | None) -> None:
        store = conv.store if conv else None
        if store is None and self.store_factory is not None and self.embedder is not None:
            try:  # made by another worker process: reach the collection through Chroma itself
                store = self.store_factory(sid)
            except Exception:
                store = None
        if store is not None:
            try:
                store.drop()
            except Exception as exc:
                log.warning("could not delete a conversation's vectors: %s", exc)
