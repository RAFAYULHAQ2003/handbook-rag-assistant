"""The ChromaDB vector store, against an in-memory stand-in for a Chroma collection (no server)."""

from datetime import date

import numpy as np
import pytest

from rag.index import HandbookIndex
from rag.retrieval.embeddings import EmbeddingCache
from rag.retrieval.vector_store import ChromaStore, VectorStoreError, collection_name, store_from_env


class FakeCollection:
    """The small part of Chroma's Collection API we use, with cosine distance like the real one."""

    name = "handbook-fake"

    def __init__(self):
        self.rows: dict[str, tuple[np.ndarray, dict]] = {}
        self.down = False
        self.calls = {"upsert": 0, "delete": 0, "query": 0}

    def _check(self):
        if self.down:
            raise ConnectionError("connection refused")

    def get(self, include=()):
        self._check()
        return {"ids": list(self.rows)}

    def upsert(self, ids, embeddings, metadatas):
        self._check()
        self.calls["upsert"] += 1
        for i, e, m in zip(ids, embeddings, metadatas, strict=True):
            self.rows[i] = (np.array(e, dtype=np.float32), m)

    def delete(self, ids):
        self._check()
        self.calls["delete"] += 1
        for i in ids:
            self.rows.pop(i, None)

    def count(self):
        self._check()
        return len(self.rows)

    def query(self, query_embeddings, n_results, include):
        self._check()
        self.calls["query"] += 1
        q = np.array(query_embeddings[0], dtype=np.float32)
        scored = sorted(((1.0 - float(v @ q), i) for i, (v, _) in self.rows.items()))[:n_results]
        return {"ids": [[i for _, i in scored]], "distances": [[d for d, _ in scored]]}


class FakeEmbedder:
    name = "fake-embedder"

    def _vec(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.random(16).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def unit(n, dim=8, seed=0):
    v = np.random.default_rng(seed).random((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_sync_adds_missing_and_removes_stale_passages():
    store = ChromaStore(FakeCollection())
    vectors = unit(3)
    assert store.sync(["a:1", "a:2", "b:1"], vectors, ["a", "a", "b"]) == (3, 0)
    assert store.sync(["a:1", "a:2", "b:1"], vectors, ["a", "a", "b"]) == (0, 0)  # repeat: nothing changes
    assert store.sync(["a:1", "c:1"], unit(2, seed=1), ["a", "c"]) == (1, 2)
    assert set(store.collection.rows) == {"a:1", "c:1"}
    assert store.collection.rows["c:1"][1] == {"doc_id": "c"}


def test_similarities_match_numpy_and_follow_the_given_order():
    store = ChromaStore(FakeCollection())
    vectors = unit(5)
    ids = [f"d:{i}" for i in range(5)]
    store.sync(ids, vectors, ["d"] * 5)
    q = vectors[2]
    assert np.allclose(store.similarities(q, ids), vectors @ q, atol=1e-5)
    assert store.similarities(q, ["d:4", "missing"])[1] == -1.0  # not in Chroma: never ranked


def test_failures_become_vector_store_errors():
    collection = FakeCollection()
    store = ChromaStore(collection)
    collection.down = True
    with pytest.raises(VectorStoreError):
        store.sync(["a:1"], unit(1), ["a"])
    with pytest.raises(VectorStoreError):
        store.similarities(unit(1)[0], ["a:1"])


def test_collection_names_are_valid_and_per_model():
    name = collection_name("BAAI/bge-small-en-v1.5")
    assert name == "handbook-baai-bge-small-en-v1.5"
    assert collection_name("BAAI/bge-small-en-v1.5 +prefix") != name


def test_store_choice_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "memory")
    assert store_from_env("m") is None
    monkeypatch.setenv("VECTOR_STORE", "elastic")
    with pytest.raises(VectorStoreError, match="unknown VECTOR_STORE"):
        store_from_env("m")
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("CHROMA_URL", "http://127.0.0.1:9")  # nothing listens there
    with pytest.raises(VectorStoreError, match="cannot connect"):
        store_from_env("m")


@pytest.fixture
def embeddings():
    return EmbeddingCache(FakeEmbedder(), None)


def test_search_through_chroma_ranks_exactly_like_memory(docs, embeddings):
    memory = HandbookIndex.build(list(docs.values()), date(2026, 11, 17), "hybrid", embeddings)
    collection = FakeCollection()
    chroma = HandbookIndex.build(list(docs.values()), date(2026, 11, 17), "hybrid", embeddings,
                                 vector_store=ChromaStore(collection))
    assert chroma.retriever.vector_store_name == "chroma handbook-fake"
    assert len(collection.rows) == len(chroma.passages)
    for q in ["How many days of annual leave do I get?", "Can I deploy to production today?", "Who is Maya?"]:
        a, b = memory.retrieve(q), chroma.retrieve(q)
        assert [h.passage.passage_id for h in a.pool] == [h.passage.passage_id for h in b.pool]
        assert [c.doc_id for c in a.context] == [c.doc_id for c in b.context]
    assert collection.calls["query"] == 3


def test_unreachable_chroma_falls_back_to_memory(docs, embeddings):
    collection = FakeCollection()
    index = HandbookIndex.build(list(docs.values()), date(2026, 11, 17), "hybrid", embeddings,
                                vector_store=ChromaStore(collection))
    expected = [h.passage.passage_id for h in index.retrieve("Who is Maya?").pool]
    collection.down = True  # Chroma goes away after start-up: each search uses the local copy
    assert [h.passage.passage_id for h in index.retrieve("Who is Maya?").pool] == expected

    down = FakeCollection()
    down.down = True  # Chroma unreachable at start-up: the index is built on memory and says so
    index = HandbookIndex.build(list(docs.values()), date(2026, 11, 17), "hybrid", embeddings,
                                vector_store=ChromaStore(down))
    assert index.retriever.vector_store_name == "memory (vector store unavailable)"
    assert index.retrieve("Who is Maya?").context


def test_error_messages_stay_short():
    from rag.retrieval.vector_store import _short
    message = _short(RuntimeError("<!DOCTYPE html>" + "<p>x</p>" * 500))
    assert len(message) < 200 and message.startswith("RuntimeError:")
