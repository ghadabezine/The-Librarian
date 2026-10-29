"""Shared helpers for the evaluation scripts."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

RESULTS = ROOT / "eval" / "results"


def load_items(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "data" / "eval" / "qa.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_gold(chunk: dict, gold: list[str]) -> bool:
    """A chunk is relevant if it belongs to one of the gold articles, in either language.

    Fixed-window chunks may span several articles: they count if any covered
    article is gold."""
    labels = chunk.get("covered_articles") or [chunk.get("article_label")]
    docs = {chunk["doc_id"], chunk.get("work_id") or chunk["doc_id"]}
    return any(f"{d}:{lab}" in gold for d in docs for lab in labels if lab)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def latency_summary(ms: list[float]) -> dict:
    return {"p50_ms": round(percentile(ms, 50), 1), "p95_ms": round(percentile(ms, 95), 1),
            "mean_ms": round(statistics.fmean(ms), 1) if ms else None, "n": len(ms)}


def history_predicate(question: str):
    from app.generation.generator import HISTORY_INTENT
    if HISTORY_INTENT.search(question):
        return None
    return lambda c: c.get("legal_status") != "historical"
