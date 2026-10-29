"""Dense embedding model wrapper (multilingual-e5-large by default).

E5 models expect "query: " / "passage: " prefixes; we add them here so the
rest of the code never has to know.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_dir: Path, device: str = "cpu", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.name = model_dir.name
        self.model = SentenceTransformer(str(model_dir), device=device)
        self.model.max_seq_length = 512
        self.batch_size = batch_size
        self.is_e5 = "e5" in self.name.lower()
        self._query_cache: dict[str, np.ndarray] = {}
        self.cache_size = 0

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self.is_e5:
            texts = ["passage: " + t for t in texts]
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 64, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if self.cache_size and text in self._query_cache:
            return self._query_cache[text]
        q = ("query: " + text) if self.is_e5 else text
        vec = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        if self.cache_size:
            if len(self._query_cache) >= self.cache_size:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[text] = vec
        return vec


@lru_cache(maxsize=4)
def get_embedder(model_dir: str, device: str = "cpu") -> Embedder:
    log.info("loading embedder %s", model_dir)
    return Embedder(Path(model_dir), device)
