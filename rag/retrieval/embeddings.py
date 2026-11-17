import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """Local ONNX embedding model via fastembed (no PyTorch, no API key).

    `query_prefix` is the instruction some models expect on queries only
    (BGE: "Represent this sentence for searching relevant passages: ");
    fastembed does not add it itself.
    """

    def __init__(self, model_name: str, query_prefix: str = "", cache_dir: Path | None = None):
        # imported lazily: loading fastembed and the model takes seconds, and BM25-only use never needs it
        from fastembed import TextEmbedding

        self.name = model_name + (" +prefix" if query_prefix else "")
        self.query_prefix = query_prefix
        cache = Path(cache_dir or os.environ.get("MODEL_CACHE_DIR") or DEFAULT_CACHE_DIR / "models")
        self._model = TextEmbedding(model_name, cache_dir=str(cache))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalise(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(next(iter(self._model.query_embed(self.query_prefix + text))), dtype=np.float32)
        return _normalise(vec[None, :])[0]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class EmbeddingCache:
    """Passage vectors on disk, keyed by model and text, so each passage is embedded once.

    An edited or uploaded passage has a new key and is embedded on first use;
    unchanged passages are read back.
    """

    def __init__(self, embedder: Embedder, path: Path | None):
        self.embedder = embedder
        self.path = path
        self._vectors: dict[str, np.ndarray] = {}
        if path is not None and path.exists():
            with np.load(path) as data:
                self._vectors = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.embedder.name}\n{text}".encode()).hexdigest()

    def get(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._vectors]
        if missing:
            new = self.embedder.embed_passages([texts[i] for i in missing])
            for i, vec in zip(missing, new, strict=True):
                self._vectors[keys[i]] = vec
            self._save()
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._vectors[k] for k in keys])

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # a unique temporary file per save, then an atomic rename: two workers saving at once,
        # or a crash mid-write, can never leave a half-written cache
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.stem, suffix=".tmp.npz")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, keys=np.array(list(self._vectors)), vectors=np.stack(list(self._vectors.values())))
            tmp.replace(self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def cache_path_for(embedder: Embedder, cache_dir: Path | None = None) -> Path:
    slug = "".join(c if c.isalnum() else "-" for c in embedder.name.lower()).strip("-")
    return Path(cache_dir or DEFAULT_CACHE_DIR) / "embeddings" / f"{slug}.npz"
