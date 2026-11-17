"""Documents uploaded through the web page, kept apart from the official handbook.

An upload is parsed with the same loaders as the handbook, stored in its own folder with a small
manifest (uploads.json), and marked `uploaded`: it can be searched and cited like any document,
but it never replaces, amends or links to an official one (see rag.validity), and the answer
step is told it is not official policy.

Every process (web worker) builds its index from the same folder; `version()` changes whenever
an upload is added or removed, so each one picks the change up on its next request.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from rag.loaders import LOADERS, EmptyDocument
from rag.manifest import load_document, slugify
from rag.models import Document

log = logging.getLogger(__name__)

MANIFEST = "uploads.json"
ALLOWED_EXTENSIONS = sorted(LOADERS)
MAX_UNZIPPED_BYTES = 50_000_000  # a .docx is a zip file: refuse "zip bombs" before parsing


class UploadError(ValueError):
    """A problem with the uploaded file, worded for the person who uploaded it."""


def _clean_title(value: str) -> str:
    """A title safe to show and to put in the prompt: no quotes, angle brackets or control characters."""
    return re.sub(r"\s+", " ", re.sub(r'[\x00-\x1f"<>`]', " ", value)).strip()[:120] or "Uploaded document"


def _check_zip(data: bytes) -> None:
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED_BYTES:
                raise UploadError("This Word file expands to more than 50 MB and cannot be read.")
    except zipfile.BadZipFile as exc:
        raise UploadError("This does not look like a valid Word (.docx) file.") from exc


class UploadStore:
    def __init__(self, folder: Path, max_bytes: int = 5_000_000, max_documents: int = 50):
        self.folder = Path(folder)
        self.max_bytes = max_bytes
        self.max_documents = max_documents

    @property
    def manifest_path(self) -> Path:
        return self.folder / MANIFEST

    def version(self) -> int:
        """Changes whenever the set of uploads changes (0 when there are none)."""
        try:
            return self.manifest_path.stat().st_mtime_ns
        except FileNotFoundError:
            return 0

    def entries(self) -> list[dict]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            log.error("uploads manifest unreadable (%s); ignoring uploads", exc)
            return []
        return [e for e in data.get("documents", []) if isinstance(e, dict) and e.get("doc_id") and e.get("path")]

    def load(self) -> list[Document]:
        """Every stored upload, parsed and marked as uploaded. A file that no longer loads is skipped."""
        docs = []
        for entry in self.entries():
            path = self.folder / Path(entry["path"]).name  # never a path outside the folder
            try:
                doc, _ = load_document(path, entry)
            except Exception as exc:  # a stored file that no longer parses must not take the site down
                log.warning("skipping upload %s: %s", entry["doc_id"], exc)
                continue
            docs.append(_mark(doc))
        return docs

    def add(self, filename: str, data: bytes, today: date) -> tuple[Document, bool]:
        """Store and parse one upload. Returns (document, created); the same file twice is stored once."""
        ext = Path(filename or "").suffix.lower()
        if ext not in LOADERS:
            raise UploadError(f"Unsupported file type. Upload one of: {', '.join(ALLOWED_EXTENSIONS)}.")
        if not data:
            raise UploadError("The file is empty.")
        if len(data) > self.max_bytes:
            raise UploadError(f"The file is larger than {self.max_bytes // 1_000_000} MB.")
        if ext == ".docx":
            _check_zip(data)

        digest = hashlib.sha256(data).hexdigest()
        doc_id = f"upload-{slugify(Path(filename).stem)[:40].strip('-')}-{digest[:8]}"
        self.folder.mkdir(parents=True, exist_ok=True)
        with self._locked():
            entries = self.entries()
            existing = next((e for e in entries if e["doc_id"] == doc_id), None)
            if existing:
                doc, _ = load_document(self.folder / existing["path"], existing)
                return _mark(doc), False
            if len(entries) >= self.max_documents:
                raise UploadError(f"The upload limit of {self.max_documents} documents is reached. Remove one first.")

            stored = self.folder / f"{doc_id}{ext}"
            fd, tmp_name = tempfile.mkstemp(dir=self.folder, prefix=".incoming-", suffix=ext)
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                try:
                    parsed, _ = load_document(tmp, {"doc_id": doc_id})
                except EmptyDocument as exc:
                    raise UploadError("No text could be found in this file (a scanned PDF needs OCR first).") from exc
                except Exception as exc:  # loaders wrap their errors, but a malformed file must never crash the page
                    raise UploadError(f"The file could not be read ({exc}).") from exc
                meta = parsed.meta
                title = meta.title if meta.title and meta.title != tmp.stem else Path(filename).stem
                entry = {
                    "path": stored.name,
                    "doc_id": doc_id,
                    "title": _clean_title(title),
                    "category": "Uploaded",
                    "owner": "Uploaded by a user",
                    "effective_date": (meta.effective_date or today).isoformat(),
                    "status": meta.status or "current",
                    "original_name": _clean_title(Path(filename).name),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": digest,
                }
                tmp.replace(stored)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            self._write(entries + [entry])
        doc, _ = load_document(stored, entry)
        return _mark(doc), True

    def remove(self, doc_id: str) -> bool:
        with self._locked():
            entries = self.entries()
            keep = [e for e in entries if e["doc_id"] != doc_id]
            if len(keep) == len(entries):
                return False
            self._write(keep)
            for e in entries:
                if e["doc_id"] == doc_id:
                    (self.folder / Path(e["path"]).name).unlink(missing_ok=True)
        return True

    def _write(self, entries: list[dict]) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=self.folder, prefix=".uploads-", suffix=".json")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"documents": entries}, f, indent=2)
            tmp.replace(self.manifest_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @contextmanager
    def _locked(self):
        """One writer at a time, across threads and worker processes."""
        self.folder.mkdir(parents=True, exist_ok=True)
        with open(self.folder / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _mark(doc: Document) -> Document:
    doc.meta.uploaded = True
    doc.meta.supersedes = None  # an upload can never replace a handbook document
    doc.meta.in_manifest = False
    return doc
