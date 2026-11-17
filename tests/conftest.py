import os
from pathlib import Path

import pytest

# Tests search vectors in memory: they must never read or write a real ChromaDB server.
os.environ["VECTOR_STORE"] = "memory"

from rag.manifest import load_corpus

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def corpus():
    """(documents by doc_id, metadata findings) for the real handbook, loaded once."""
    documents, findings = load_corpus(CORPUS_DIR)
    return {d.doc_id: d for d in documents}, findings


@pytest.fixture(scope="session")
def docs(corpus):
    return corpus[0]
