import json
import re
from datetime import date
from pathlib import Path

from rag.loaders import SUPPORTED_EXTENSIONS, LoaderError, build_text, parse_file
from rag.models import Document, DocumentMeta, MetadataFinding

FIELDS = ("doc_id", "title", "category", "owner", "effective_date", "status", "supersedes")
FORMATS = {".md": "md", ".markdown": "md", ".pdf": "pdf", ".docx": "docx", ".csv": "csv",
           ".txt": "txt", ".html": "html", ".htm": "html"}


def slugify(value: str) -> str:
    """URL-safe document id: 'My Policy (v2).pdf' stem -> 'my-policy-v2'."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def read_manifest(corpus_dir: Path) -> dict[str, dict]:
    """Manifest entries keyed by file name."""
    path = corpus_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest.json is not valid JSON: {exc}") from exc
    return {entry["path"]: entry for entry in data.get("documents", []) if entry.get("path")}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _norm(value) -> str | None:
    return None if value in (None, "") else str(value).strip()


def load_document(path: Path, manifest_entry: dict | None = None) -> tuple[Document, list[MetadataFinding]]:
    """Load one file. Raises LoaderError if the file cannot be read."""
    parsed = parse_file(path)
    file_meta = dict(parsed.meta)
    if parsed.title:
        file_meta.setdefault("title", parsed.title)

    entry = manifest_entry or {}
    doc_id = _norm(entry.get("doc_id")) or slugify(_norm(file_meta.get("doc_id")) or path.stem)
    merged: dict[str, str | None] = {}
    findings: list[MetadataFinding] = []
    for name in FIELDS:
        m_val, f_val = _norm(entry.get(name)), _norm(file_meta.get(name))
        merged[name] = m_val if m_val is not None else f_val
        if entry and m_val is not None and f_val is not None and m_val.lower() != f_val.lower():
            findings.append(MetadataFinding(doc_id, name, m_val, f_val))
        elif entry and m_val is None and f_val is not None:
            findings.append(MetadataFinding(doc_id, name, None, f_val))

    effective = _parse_date(merged["effective_date"])
    if merged["effective_date"] and effective is None:
        findings.append(MetadataFinding(doc_id, "effective_date", merged["effective_date"], "unparseable"))

    title = merged["title"] or path.stem
    text, blocks = build_text(parsed.blocks, title=title)
    meta = DocumentMeta(
        doc_id=doc_id,
        title=title,
        path=path.name,
        format=FORMATS.get(path.suffix.lower(), path.suffix.lower().lstrip(".")),
        category=merged["category"],
        owner=merged["owner"],
        effective_date=effective,
        status=(merged["status"] or "").lower() or None,
        supersedes=merged["supersedes"],
        in_manifest=bool(entry),
    )
    return Document(meta=meta, text=text, blocks=blocks), findings


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_corpus(corpus_dir: Path) -> tuple[list[Document], list[MetadataFinding]]:
    """Load every manifest document, plus any supported file the manifest does not list."""
    manifest = read_manifest(corpus_dir)
    documents: list[Document] = []
    findings: list[MetadataFinding] = []
    seen: set[str] = set()

    def add(path: Path, entry: dict | None):
        label = (entry or {}).get("doc_id") or path.name
        try:
            doc, doc_findings = load_document(path, entry)
        except LoaderError as exc:
            findings.append(MetadataFinding(label, "load_error", path.name, str(exc)))
            return
        findings.extend(doc_findings)
        if doc.doc_id in seen:
            findings.append(MetadataFinding(doc.doc_id, "duplicate_doc_id", None, path.name))
            return
        seen.add(doc.doc_id)
        documents.append(doc)

    for name, entry in manifest.items():
        path = corpus_dir / name
        if not _inside(path, corpus_dir):
            findings.append(MetadataFinding(entry.get("doc_id", name), "path_outside_corpus", name, None))
        elif not path.is_file():
            findings.append(MetadataFinding(entry.get("doc_id", name), "path", name, None))
        else:
            add(path, entry)

    for path in sorted(corpus_dir.iterdir()):
        if (path.is_file() and path.name not in manifest and not path.name.startswith(".")
                and path.suffix.lower() in SUPPORTED_EXTENSIONS):
            add(path, None)
    return documents, findings
