"""Uploads belong to one conversation: checked, chunked, embedded for it alone, and deleted when it ends."""

import io
import time
import zipfile
from datetime import date

import pytest

from rag.uploads import SessionUploads, UploadError, valid_session_id

TODAY = date(2026, 11, 17)
NOTE = b"# Team Offsite\n\nThe platform team offsite is in Lisbon on 2027-03-10.\n"
A, B = "a" * 32, "b" * 32  # two conversations


@pytest.fixture
def uploads(tmp_path):
    return SessionUploads(tmp_path / "uploads", max_bytes=10_000, max_documents=3, ttl_seconds=3600)


def test_conversation_ids_are_checked():
    assert valid_session_id(A) == A and valid_session_id(A.upper()) == A
    for bad in ("", None, "abc", "../" + "a" * 29, "z" * 32):
        assert valid_session_id(bad) is None


def test_an_upload_is_chunked_and_searchable_in_its_conversation_only(uploads):
    doc, created = uploads.add(A, "Team Offsite.md", NOTE, TODAY)
    assert created and doc.doc_id.startswith("upload-team-offsite-") and doc.meta.uploaded
    conv = uploads.get(A)
    assert doc.doc_id in conv.docs and conv.passages_by_doc[doc.doc_id]
    hits = conv.retriever.search("Where is the offsite?", k=3).hits
    assert hits and hits[0].passage.doc_id == doc.doc_id
    assert uploads.get(B) is None and uploads.documents(B) == []  # another conversation sees nothing


def test_same_file_twice_is_stored_once(uploads):
    first, _ = uploads.add(A, "offsite.md", NOTE, TODAY)
    again, created = uploads.add(A, "offsite.md", NOTE, TODAY)
    assert not created and again.doc_id == first.doc_id and len(uploads.documents(A)) == 1


@pytest.mark.parametrize("name, data, message", [
    ("notes.exe", b"MZ", "Unsupported file type"),
    ("noext", b"hello", "Unsupported file type"),
    ("empty.md", b"", "empty"),
    ("big.txt", b"x" * 10_001, "larger than"),
    ("broken.docx", b"not a zip", "valid Office file"),
    ("broken.xlsx", b"not a zip", "valid Office file"),
    ("blank.txt", b"   \n\n", "No text could be found"),
])
def test_bad_uploads_are_refused_clearly(uploads, name, data, message):
    with pytest.raises(UploadError, match=message):
        uploads.add(A, name, data, TODAY)
    assert uploads.documents(A) == []


def test_a_scanned_pdf_explains_ocr(uploads):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # a page with no text layer, like a scan
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(UploadError, match="needs OCR"):
        uploads.add(A, "scan.pdf", buffer.getvalue(), TODAY)


def test_zip_bombs_are_refused(uploads, monkeypatch):
    monkeypatch.setattr("rag.uploads.MAX_UNZIPPED_BYTES", 1000)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "a" * 5000)
    with pytest.raises(UploadError, match="expands"):
        uploads.add(A, "bomb.docx", buffer.getvalue(), TODAY)


def test_per_conversation_limit(uploads):
    for i in range(3):
        uploads.add(A, f"n{i}.txt", f"Note number {i} about the offsite.".encode(), TODAY)
    with pytest.raises(UploadError, match="already has 3 documents"):
        uploads.add(A, "n4.txt", b"One more note about the offsite.", TODAY)
    uploads.add(B, "n4.txt", b"One more note about the offsite.", TODAY)  # other conversations are separate


def test_remove_takes_a_document_out(uploads):
    doc, _ = uploads.add(A, "offsite.md", NOTE, TODAY)
    assert uploads.remove(A, doc.doc_id) and uploads.documents(A) == []
    assert not uploads.remove(A, doc.doc_id) and not uploads.remove(B, doc.doc_id)


def test_new_conversation_deletes_files_and_vectors(tmp_path):
    dropped = []

    class Store:
        name = "chroma conversation"

        def __init__(self, sid):
            self.sid = sid

        def sync(self, ids, vectors, doc_ids):
            return len(ids), 0

        def drop(self):
            dropped.append(self.sid)

    from rag.uploads import Conversation
    uploads = SessionUploads(tmp_path / "u")
    uploads.add(A, "offsite.md", NOTE, TODAY)
    uploads._conversations[A] = Conversation(A, tmp_path / "u" / A, store=Store(A))  # as if vectors were in Chroma
    uploads.clear(A)
    assert dropped == [A] and not (tmp_path / "u" / A).exists() and uploads.get(A) is None


def test_idle_conversations_are_deleted(uploads):
    uploads.add(A, "offsite.md", NOTE, TODAY)
    uploads.add(B, "offsite.md", NOTE, TODAY)
    assert uploads.sweep(now=time.time() + 3601) == 2 and uploads.documents(A) == []
    uploads.add(B, "offsite.md", NOTE, TODAY)
    assert uploads.sweep(now=time.time() + 60) == 0 and uploads.documents(B)


def test_another_worker_sees_the_conversation_from_its_folder(uploads, tmp_path):
    doc, _ = uploads.add(A, "offsite.md", NOTE, TODAY)
    other_worker = SessionUploads(tmp_path / "uploads")
    assert [d.doc_id for d in other_worker.documents(A)] == [doc.doc_id]


def test_an_upload_cannot_claim_to_replace_a_handbook_document(uploads):
    text = (b"---\ndoc_id: hr-leave-policy-v2\ntitle: Leave Policy v3\nsupersedes: hr-leave-policy-v2\n---\n\n"
            b"# Leave Policy v3\n\nEveryone gets 60 days of annual leave.\n")
    doc, _ = uploads.add(A, "leave.md", text, TODAY)
    assert doc.doc_id.startswith("upload-") and doc.meta.supersedes is None and doc.meta.uploaded


def test_titles_are_made_safe(uploads):
    doc, _ = uploads.add(A, 'x" onload="y.md', b'---\ntitle: Evil" <script>\n---\n\nSome text about travel.\n', TODAY)
    assert '"' not in doc.meta.title and "<" not in doc.meta.title


def test_conversation_documents_join_the_handbook_search(docs, uploads):
    from rag.index import HandbookIndex
    index = HandbookIndex.build(list(docs.values()), TODAY, mode="bm25")
    doc, _ = uploads.add(A, "offsite.md", NOTE, TODAY)
    retrieval = index.retrieve("Where is the platform team offsite?", uploads=uploads.get(A))
    assert retrieval.context[0].doc_id == doc.doc_id and doc.doc_id in retrieval.docs
    assert retrieval.document(doc.doc_id, index) is retrieval.docs[doc.doc_id]
    plain = index.retrieve("Where is the platform team offsite?")
    assert doc.doc_id not in {c.doc_id for c in plain.context}  # without the conversation, nothing leaks
