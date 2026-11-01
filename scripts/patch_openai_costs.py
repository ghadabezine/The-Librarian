"""Backfill cost_usd into an existing generation eval run.

Reads the run's .jsonl, recomputes each row's cost from its own token counts
using the current gpt-4o-mini list prices, and writes the aggregate
mean_cost_usd into the matching _summary.json. No API calls, no re-run.

Usage:
    python scripts/patch_openai_costs.py eval/results/generation_openai
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Current OpenAI list prices, USD per million tokens (2026-10).
# Versioned model names (e.g. gpt-4o-mini-2024-07-18) match by prefix.
PRICES = {
    "gpt-4o":       (2.50, 10.00),
    "gpt-4o-mini":  (0.15,  0.60),
    "gpt-4.1":      (2.00,  8.00),
    "gpt-4.1-mini": (0.40,  1.60),
    "gpt-4.1-nano": (0.10,  0.40),
}


def price_for(model: str) -> tuple[float, float]:
    # Longest-key-first so "gpt-4o-mini-2024-07-18" matches "gpt-4o-mini",
    # not the shorter "gpt-4o" prefix.
    for key, (pin, pout) in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
        if model == key or model.startswith(key):
            return pin, pout
    raise SystemExit(f"no price for model {model!r}; add it to PRICES")


def main(prefix: str) -> None:
    jsonl = Path(f"{prefix}.jsonl")
    summary_path = Path(f"{prefix}_summary.json")
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}")
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")

    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    total = 0.0
    patched = 0
    for r in rows:
        u = r.setdefault("usage", {})
        model = u.get("model") or "gpt-4o-mini"
        pin, pout = price_for(model)
        cost = u.get("input_tokens", 0) / 1e6 * pin + u.get("output_tokens", 0) / 1e6 * pout
        if abs(u.get("cost_usd", 0.0) - cost) > 1e-12:
            u["cost_usd"] = cost
            patched += 1
        total += cost

    with jsonl.open("w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mean = total / len(rows) if rows else 0.0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "overall" in summary:
        summary["overall"]["mean_cost_usd"] = mean

    def mean_cost(subset):
        return sum(
            (r.get("usage") or {}).get("cost_usd", 0.0) for r in subset
        ) / len(subset) if subset else None

    for lang, block in (summary.get("by_language") or {}).items():
        block["mean_cost_usd"] = mean_cost([r for r in rows if r.get("lang") == lang])
    for cat, block in (summary.get("by_category") or {}).items():
        block["mean_cost_usd"] = mean_cost([r for r in rows if r.get("cat") == cat])

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"patched   {patched} / {len(rows)} rows in {jsonl}")
    print(f"updated   {summary_path}")
    print(f"total     ${total:.6f}")
    print(f"mean/q    ${mean:.6f}")
    print(f"per 1k    ${mean * 1000:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/results/generation_openai")