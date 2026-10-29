"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3, multilingual)."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_dir: Path, device: str = "cpu", max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.name = model_dir.name
        self.model = CrossEncoder(str(model_dir), device=device, max_length=max_length)

    def score(self, query: str, passages: list[str], batch_size: int = 8) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        # bge-reranker outputs a logit; the CrossEncoder applies a sigmoid for
        # single-label models, so scores are in [0, 1].
        scores = self.model.predict([(query, p) for p in passages], batch_size=batch_size,
                                    show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


@lru_cache(maxsize=2)
def get_reranker(model_dir: str, device: str = "cpu", max_length: int = 512) -> Reranker:
    log.info("loading reranker %s", model_dir)
    return Reranker(Path(model_dir), device, max_length)
