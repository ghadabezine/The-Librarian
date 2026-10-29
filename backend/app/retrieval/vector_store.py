"""Flat (exact) vector index persisted as .npy + chunk metadata as JSONL.

Why not a vector database: the corpus is a few thousand chunks. An exact
dot-product over a (n x 1024) float32 matrix takes well under a millisecond,
has perfect recall, needs no server, and the index is two files that are easy
to rebuild, diff and ship in a Docker volume. Metadata filtering is a boolean
mask over the chunk list. If the corpus grew to millions of chunks we would
switch to FAISS-HNSW or pgvector; the interface below would not change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class VectorIndex:
    vectors: np.ndarray          # (n, d) L2-normalised float32
    chunk_ids: list[str]
    model_name: str

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "vectors_meta.json").write_text(json.dumps(
            {"model": self.model_name, "dim": int(self.vectors.shape[1]), "chunk_ids": self.chunk_ids},
            ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        meta = json.loads((index_dir / "vectors_meta.json").read_text(encoding="utf-8"))
        vectors = np.load(index_dir / "vectors.npy")
        if len(meta["chunk_ids"]) != vectors.shape[0]:
            raise ValueError("vector index is inconsistent with its metadata; rebuild it")
        return cls(vectors, meta["chunk_ids"], meta["model"])

    def search(self, query_vec: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        scores = self.vectors @ query_vec
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


def build_mask(chunks: list[dict], predicate: Callable[[dict], bool] | None) -> np.ndarray | None:
    if predicate is None:
        return None
    return np.fromiter((predicate(c) for c in chunks), dtype=bool, count=len(chunks))
