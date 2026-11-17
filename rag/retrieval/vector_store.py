"""Where passage vectors live for meaning search: in a ChromaDB server (default) or in memory.

Either way the vectors are made by our own embedding model (bge-small, see embeddings.py), so
switching stores never changes what is compared, only where. Keyword search, fusion and every
validity rule stay in rag.retrieval.search.

ChromaDB runs as its own server (`docker compose` starts one), shared by every web worker. It is
used in client/server mode only: the embedded mode keeps its data in SQLite, which several worker
processes must not write at once. The server has no authentication, so it must never be exposed
outside the compose network.

If Chroma cannot be reached, search falls back to the in-memory copy of the vectors (169 passages
take about 260 KB), so answers keep working; the failure is logged and shown by /api/health.
VECTOR_STORE=memory skips Chroma entirely (the tests do this).
"""

import logging
import os
import re

import numpy as np

log = logging.getLogger(__name__)

BATCH = 500  # passages per upsert call


class VectorStoreError(RuntimeError):
    pass


def _short(exc: Exception) -> str:
    """One readable line: a wrong address can answer with a whole HTML page."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:160]}" + ("..." if len(text) > 160 else "")


def collection_name(model_name: str, prefix: str = "handbook") -> str:
    """One collection per embedding model, so vectors from two models never mix.
    Chroma allows 3-512 characters of [a-zA-Z0-9._-], starting and ending with a letter or digit."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", f"{prefix}-{model_name}".lower()).strip("-._")
    return slug[:200]


class ChromaStore:
    """Passage vectors in a ChromaDB collection, keyed by passage id (a content hash).

    `sync()` makes the collection hold exactly the given passages: new ones are added, ones no
    longer in the handbook (a removed upload, an edited document) are deleted. Ids are content
    hashes, so running it again, or from several workers at once, changes nothing.
    """

    def __init__(self, collection, url: str = ""):
        self.collection = collection
        self.name = f"chroma {collection.name}"
        self.url = url

    @classmethod
    def connect(cls, url: str, model_name: str, prefix: str = "handbook") -> "ChromaStore":
        try:
            from urllib.parse import urlparse

            import chromadb

            parts = urlparse(url if "://" in url else f"http://{url}")
            client = chromadb.HttpClient(host=parts.hostname or "localhost", port=parts.port or 8000,
                                         ssl=parts.scheme == "https")
            collection = client.get_or_create_collection(
                collection_name(model_name, prefix),
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,  # we always pass our own vectors; never let Chroma embed
            )
        except Exception as exc:  # unreachable server, bad URL, version mismatch
            raise VectorStoreError(f"cannot connect to ChromaDB at {url}: {_short(exc)}") from exc
        return cls(collection, url)

    def sync(self, ids: list[str], vectors: np.ndarray, doc_ids: list[str]) -> tuple[int, int]:
        """Make the collection hold exactly these passages. Returns (added, removed)."""
        try:
            existing = set(self.collection.get(include=[])["ids"])
            wanted = dict(zip(ids, range(len(ids)), strict=True))
            missing = [i for pid, i in wanted.items() if pid not in existing]
            for start in range(0, len(missing), BATCH):
                rows = missing[start:start + BATCH]
                self.collection.upsert(ids=[ids[i] for i in rows], embeddings=vectors[rows].tolist(),
                                       metadatas=[{"doc_id": doc_ids[i]} for i in rows])
            stale = sorted(existing - wanted.keys())
            for start in range(0, len(stale), BATCH):
                self.collection.delete(ids=stale[start:start + BATCH])
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB sync failed: {_short(exc)}") from exc
        return len(missing), len(stale)

    def similarities(self, query: np.ndarray, ids: list[str]) -> np.ndarray:
        """Cosine similarity of the query to each passage in `ids` (-1 for any Chroma did not return)."""
        try:
            total = self.collection.count()
            if total == 0:
                return np.full(len(ids), -1.0)
            result = self.collection.query(query_embeddings=[query.tolist()], n_results=total, include=["distances"])
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {_short(exc)}") from exc
        score = {pid: 1.0 - float(d) for pid, d in zip(result["ids"][0], result["distances"][0], strict=True)}
        return np.array([score.get(pid, -1.0) for pid in ids])


def store_from_env(model_name: str) -> ChromaStore | None:
    """The ChromaDB store (server at CHROMA_URL), or None when VECTOR_STORE=memory.
    Raises VectorStoreError if Chroma cannot be reached."""
    kind = os.environ.get("VECTOR_STORE", "").strip().lower() or "chroma"
    if kind == "memory":
        return None
    if kind != "chroma":
        raise VectorStoreError(f"unknown VECTOR_STORE {kind!r}; use memory or chroma")
    # 8001, not Chroma's usual 8000: Django's development server already uses 8000 locally
    url = os.environ.get("CHROMA_URL", "").strip() or "http://localhost:8001"
    prefix = os.environ.get("CHROMA_COLLECTION_PREFIX", "").strip() or "handbook"
    return ChromaStore.connect(url, model_name, prefix)
