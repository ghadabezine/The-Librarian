"""Retrieval orchestrator: BM25, dense, hybrid (RRF) and hybrid + rerank."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.config import Settings
from .bm25 import BM25Index
from .fusion import rrf
from .query import QueryInfo, analyse
from .vector_store import VectorIndex, build_mask

log = logging.getLogger(__name__)

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
# A question that names a law other than the Code ("القانون عدد 94 لسنة 1998", "loi n°98-94").
OTHER_LAW = re.compile(r"(?:قانون|القانون|أمر|الأمر|قرار|loi|décret|arrêté|law|decree)\s+(?:عدد|n\s*°|no\.?|number)?\s*\d",
                       re.IGNORECASE)


@dataclass
class Hit:
    chunk: dict
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None
    via_article_ref: bool = False


@dataclass
class RetrievalResult:
    query: QueryInfo
    hits: list[Hit]
    mode: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    reranked: bool = False


def load_chunks(index_dir: Path) -> list[dict]:
    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Retriever:
    def __init__(self, settings: Settings, chunks: list[dict] | None = None,
                 vector_index: VectorIndex | None = None, embedder=None, reranker=None):
        self.s = settings
        self.chunks = chunks if chunks is not None else load_chunks(settings.index_dir)
        self.by_id = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25Index([c["search_text"] for c in self.chunks])
        self._vindex = vector_index
        self._embedder = embedder
        self._reranker = reranker

    # Lazy model loading keeps BM25-only use (tests, ablations) light.
    @property
    def vindex(self) -> VectorIndex:
        if self._vindex is None:
            self._vindex = VectorIndex.load(self.s.index_dir)
            ids = [c["chunk_id"] for c in self.chunks]
            if self._vindex.chunk_ids != ids:
                raise ValueError("vector index does not match chunks.jsonl; run scripts/build_index.py")
        return self._vindex

    @property
    def embedder(self):
        if self._embedder is None:
            from .embedder import get_embedder
            self._embedder = get_embedder(str(self.s.models_dir / self.s.embedding_model), self.s.device)
            self._embedder.cache_size = self.s.query_cache_size
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from .reranker import get_reranker
            self._reranker = get_reranker(str(self.s.models_dir / self.s.reranker_model), self.s.device)
        return self._reranker

    def _article_lookup(self, info: QueryInfo, mask: np.ndarray | None) -> list[int]:
        """Chunks of the articles cited in the question ("الفصل 5", "article 32 bis").

        Article numbers restart in every law, so unless the question names
        another text the lookup is limited to the Personal Status Code. Chunks
        in the question's language come first."""
        if not info.article_refs:
            return []
        other_law = OTHER_LAW.search(info.text)
        out = []
        for i, c in enumerate(self.chunks):
            if mask is not None and not mask[i]:
                continue
            if c.get("chunk_type") != "article" or c.get("article_label") not in info.article_refs:
                continue
            if not other_law and (c.get("work_id") or c["doc_id"]) != "csp":
                continue
            out.append(i)
        return sorted(out, key=lambda i: self.chunks[i].get("language") != info.language)

    def retrieve(self, question: str, mode: str | None = None, top_k: int | None = None,
                 filters: dict | None = None, predicate: Callable[[dict], bool] | None = None) -> RetrievalResult:
        mode = mode or self.s.retrieval_mode
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        top_k = top_k or self.s.top_k
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        info = analyse(question)

        if filters:
            def pred(c, f=filters, p=predicate):
                return all(c.get(k) in (v if isinstance(v, list) else [v]) for k, v in f.items()) and (p is None or p(c))
            predicate = pred
        mask = build_mask(self.chunks, predicate)

        bm25_hits: list[tuple[int, float]] = []
        dense_hits: list[tuple[int, float]] = []
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            bm25_hits = self.bm25.search(info.keyword_query, self.s.candidate_k, mask)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            qv = self.embedder.embed_query(info.text)
            timings["embed"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dense_hits = self.vindex.search(qv, self.s.candidate_k, mask)
            timings["dense"] = (time.perf_counter() - t) * 1000

        bm25_rank = {i: r for r, (i, _) in enumerate(bm25_hits, 1)}
        dense_rank = {i: r for r, (i, _) in enumerate(dense_hits, 1)}
        article_ids = self._article_lookup(info, mask)

        if mode == "bm25":
            ranked = bm25_hits
        elif mode == "dense":
            ranked = dense_hits
        else:
            ranked = rrf([[i for i, _ in bm25_hits], [i for i, _ in dense_hits]], k=self.s.rrf_k)

        # Explicit "article N" references are answered by structure, not by
        # similarity: put those chunks first.
        if article_ids:
            top = max((s for _, s in ranked), default=1.0)
            seen = set(article_ids)
            ranked = [(i, top + 1.0) for i in article_ids] + [(i, s) for i, s in ranked if i not in seen]

        reranked = False
        rerank_scores: dict[int, float] = {}
        if mode == "hybrid_rerank" and ranked:
            confident = (self.s.skip_rerank_when_confident and bm25_hits and dense_hits
                         and bm25_hits[0][0] == dense_hits[0][0])
            if not confident:
                t = time.perf_counter()
                cand = [i for i, _ in ranked[: self.s.rerank_k]]
                scores = self.reranker.score(info.text, [self.chunks[i]["embed_text"] for i in cand])
                rerank_scores = dict(zip(cand, map(float, scores)))
                pinned = [i for i in cand if i in set(article_ids)]
                rest = sorted((i for i in cand if i not in set(article_ids)), key=lambda i: -rerank_scores[i])
                ranked = [(i, rerank_scores[i]) for i in pinned + rest]
                timings["rerank"] = (time.perf_counter() - t) * 1000
                reranked = True

        if self.s.collapse_translations:
            seen_keys, collapsed = set(), []
            for i, sc in ranked:
                c = self.chunks[i]
                key = (c.get("work_id") or c["doc_id"], c.get("chunk_type"), c.get("article_label"), c.get("part"))
                if c.get("article_label") and key in seen_keys:
                    continue
                seen_keys.add(key)
                collapsed.append((i, sc))
            ranked = collapsed

        hits = [Hit(chunk=self.chunks[i], score=float(s), bm25_rank=bm25_rank.get(i),
                    dense_rank=dense_rank.get(i), rerank_score=rerank_scores.get(i),
                    via_article_ref=i in set(article_ids))
                for i, s in ranked[:top_k]]
        timings["total"] = (time.perf_counter() - t0) * 1000
        return RetrievalResult(info, hits, mode, timings, reranked)
