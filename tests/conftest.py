from pathlib import Path

import pytest

from rag.manifest import load_corpus

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def corpus():
    """(documents by doc_id, metadata findings) for the real handbook, loaded once."""
    documents, findings = load_corpus(CORPUS_DIR)
    return {d.doc_id: d for d in documents}, findings


@pytest.fixture(scope="session")
def docs(corpus):
    return corpus[0]
