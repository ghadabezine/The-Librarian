"""Reciprocal Rank Fusion (Cormack et al., 2009)."""
from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Fuse several ranked lists of document indices.

    score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
