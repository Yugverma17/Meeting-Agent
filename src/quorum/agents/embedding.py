"""Local text embeddings.

Two implementations behind one protocol:

- `FastEmbedEmbedder` is the real one. fastembed runs BGE through ONNX Runtime
  (~100 MB) instead of sentence-transformers' torch stack (~2.5 GB) - a
  deliberate choice for a 7.6 GB machine with no GPU.
- `LexicalEmbedder` is a dependency-free hashing fallback. It exists so tests
  run in milliseconds without downloading a model, and so the pipeline degrades
  to something workable if the ONNX model cannot be fetched.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_TOKEN_RE = re.compile(r"[a-z0-9']+")


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float array of L2-normalised row vectors."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LexicalEmbedder:
    """Hashed bag-of-words with sublinear term weighting.

    Not semantic - it cannot tell that "deadline" and "due date" are related.
    Good enough to find topic boundaries in a transcript, where speakers
    generally do change vocabulary when the subject changes, and it makes the
    test suite deterministic and instant.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                matrix[row, self._bucket(token)] += 1.0
        # Sublinear scaling stops one repeated filler word dominating a turn.
        matrix = np.log1p(matrix)
        return _normalise(matrix)


class FastEmbedEmbedder:
    """BGE-small via ONNX. Downloads ~130 MB on first use, then caches."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            log.info("Loading embedding model %s (first run downloads it)", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = np.array(list(self._load().embed(texts)), dtype=np.float32)
        return _normalise(vectors)


def get_embedder(prefer_local_model: bool = True) -> Embedder:
    """Best available embedder, degrading rather than failing.

    A missing ONNX model is not worth crashing a pipeline run over - lexical
    segmentation still produces usable boundaries.
    """
    if prefer_local_model:
        try:
            embedder = FastEmbedEmbedder()
            embedder.embed(["warm up"])
            return embedder
        except Exception as exc:  # noqa: BLE001 - fastembed raises varied types
            log.warning("Falling back to lexical embeddings (%s): %s", type(exc).__name__, exc)
    return LexicalEmbedder()
