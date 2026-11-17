import json
from datetime import date

import pytest

from rag.manifest import load_corpus, load_document, read_manifest
from tests.conftest import CORPUS_DIR


def test_every_manifest_document_loads(docs):
    manifest = read_manifest(CORPUS_DIR)
    assert {e["doc_id"] for e in manifest.values()} == set(docs)
    assert all(d.meta.in_manifest for d in docs.values())


def test_formats_detected(docs):
    assert {d.meta.format for d in docs.values()} == {"md", "pdf", "docx", "csv", "txt", "html"}


def test_metadata_from_manifest(docs):
    meta = docs["eng-deploy-runbook-2023"].meta
    assert meta.status == "deprecated"
    assert meta.effective_date == date(2023, 4, 1)
    assert meta.owner == "Platform Team"


def test_csv_and_txt_get_metadata_only_from_manifest(docs):
    meta = docs["fin-per-diem-table"].meta
    assert meta.title == "Per-Diem Table"
    assert meta.effective_date == date(2025, 1, 1)


def test_supersedes_recorded(docs):
    assert docs["hr-leave-policy-v2"].meta.supersedes == "hr-leave-policy-v1"


def test_declared_status_kept_as_is(docs):
    # v1 is still "current" in the manifest; deciding it is superseded is rag.validity's job
    assert docs["hr-leave-policy-v1"].meta.status == "current"
    assert docs["hr-parental-leave-policy-2027"].meta.effective_date == date(2027, 1, 1)


def test_disagreement_recorded_as_finding(corpus):
    _, findings = corpus
    titles = [f for f in findings if f.doc_id == "ref-handbook-index" and f.field == "title"]
    assert titles and titles[0].file_value == "Northgate Robotics Handbook Index"


def test_file_not_in_manifest_uses_its_own_metadata(tmp_path):
    path = tmp_path / "hr-parking.md"
    path.write_text(
        "---\ndoc_id: hr-parking\ntitle: Parking Policy\neffective_date: 2026-05-01\n"
        "status: current\nsupersedes: hr-parking-old\n---\n\n# Parking Policy\n\nVisitors park on level 2.\n"
    )
    doc, findings = load_document(path)
    assert doc.doc_id == "hr-parking"
    assert doc.meta.title == "Parking Policy"
    assert doc.meta.effective_date == date(2026, 5, 1)
    assert doc.meta.supersedes == "hr-parking-old"
    assert not doc.meta.in_manifest
    assert findings == []


def test_file_without_any_metadata_falls_back_to_file_name(tmp_path):
    path = tmp_path / "rooms.txt"
    path.write_text("Room Atlas seats eight people.\n")
    doc, _ = load_document(path)
    assert doc.doc_id == "rooms"
    assert doc.meta.title == "rooms"
    assert doc.meta.effective_date is None


def test_missing_manifest_file_is_reported(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"documents": [{"doc_id": "gone", "path": "gone.md", "title": "Gone"}]}
    ))
    (tmp_path / "extra.md").write_text("# Extra\n\nNot in the manifest.\n")
    documents, findings = load_corpus(tmp_path)
    assert [d.doc_id for d in documents] == ["extra"]
    assert any(f.doc_id == "gone" and f.field == "path" for f in findings)


# ---------- robustness ----------

def _write_manifest(root, entries):
    (root / "manifest.json").write_text(json.dumps({"documents": entries}))


def test_one_bad_file_does_not_stop_the_corpus(tmp_path):
    _write_manifest(tmp_path, [{"doc_id": "bad", "path": "bad.pdf"}, {"doc_id": "good", "path": "good.md"}])
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    (tmp_path / "good.md").write_text("Good text.\n")
    documents, findings = load_corpus(tmp_path)
    assert [d.doc_id for d in documents] == ["good"]
    assert any(f.doc_id == "bad" and f.field == "load_error" for f in findings)


def test_hidden_and_unsupported_files_skipped(tmp_path):
    (tmp_path / "._copy.md").write_bytes(b"\x00\x05binary resource fork")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "real.md").write_text("Real text.\n")
    documents, findings = load_corpus(tmp_path)
    assert [d.doc_id for d in documents] == ["real"]
    assert findings == []


def test_duplicate_doc_id_reported(tmp_path):
    _write_manifest(tmp_path, [{"doc_id": "policy", "path": "policy.md"}])
    (tmp_path / "policy.md").write_text("Original.\n")
    (tmp_path / "copy.md").write_text("---\ndoc_id: policy\n---\n\nCopy.\n")
    documents, findings = load_corpus(tmp_path)
    assert [d.meta.path for d in documents] == ["policy.md"]
    assert any(f.field == "duplicate_doc_id" for f in findings)


def test_manifest_path_outside_corpus_rejected(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (tmp_path / "secret.md").write_text("Secret.\n")
    _write_manifest(corpus, [{"doc_id": "escape", "path": "../secret.md"}])
    documents, findings = load_corpus(corpus)
    assert documents == []
    assert any(f.field == "path_outside_corpus" for f in findings)


def test_doc_id_from_file_name_is_url_safe(tmp_path):
    path = tmp_path / "My Policy (v2).md"
    path.write_text("Text.\n")
    doc, _ = load_document(path)
    assert doc.doc_id == "my-policy-v2"


def test_unparseable_date_reported(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("---\ndoc_id: p\neffective_date: next spring\n---\n\nText.\n")
    doc, findings = load_document(path)
    assert doc.meta.effective_date is None
    assert any(f.field == "effective_date" for f in findings)


def test_invalid_manifest_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ValueError, match="manifest.json"):
        load_corpus(tmp_path)
