"""Offline analysis of eval/results/rerank_scores.jsonl.

1. Rerank depth: Recall@k when only the first N fused candidates are reranked
   (exact, because pairs are scored independently), with and without skipping
   the reranker when BM25 and dense agree on the top hit. Rerank latency is
   estimated as the measured 20-pair latency scaled by N/20 (cost is linear in
   the number of pairs).
2. Abstention threshold: false abstentions on answerable questions versus
   abstentions on unanswerable ones, for candidate thresholds on the best
   reranker score.

    python eval/simulate_rerank.py
"""
from __future__ import annotations

import json
import statistics

from common import RESULTS, percentile


def ranked(row, depth, skip_confident):
    fused = row["fused"]
    pinned = [c for c in fused if c["via_article_ref"]]
    if skip_confident and fused and fused[0]["bm25_rank"] == 1 and fused[0]["dense_rank"] == 1:
        return fused, False
    head = [c for c in fused[:depth] if not c["via_article_ref"]]
    head.sort(key=lambda c: -(c["rerank"] or 0))
    rest = [c for c in fused[depth:] if not c["via_article_ref"]]
    return pinned + head + rest, True


def main() -> None:
    rows = [json.loads(l) for l in open(RESULTS / "rerank_scores.jsonl", encoding="utf-8")]
    ans = [r for r in rows if r["answerable"]]
    lines = ["| Rerank depth | Skip if BM25 & dense agree | R@1 | R@3 | R@5 | Reranked share | Est. rerank p50 (ms) | Est. rerank p95 (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    out = {"depth": [], "threshold": []}
    for depth in (20, 10, 8, 5, 0):
        for skip in (False, True):
            if depth == 0 and skip:
                continue
            hits = {1: 0, 3: 0, 5: 0}
            lat, n_rr = [], 0
            for r in ans:
                order, did = ranked(r, depth, skip) if depth else (r["fused"], False)
                n_rr += did
                for k in hits:
                    hits[k] += any(c["gold"] for c in order[:k])
                lat.append((r["rerank_ms"] or 0) * depth / 20 if did else 0.0)
            rec = {k: round(v / len(ans), 4) for k, v in hits.items()}
            row = {"depth": depth, "skip_confident": skip, **{f"recall@{k}": v for k, v in rec.items()},
                   "reranked_share": round(n_rr / len(ans), 3),
                   "rerank_p50_ms": round(percentile(lat, 50)), "rerank_p95_ms": round(percentile(lat, 95))}
            out["depth"].append(row)
            lines.append(f"| {depth or 'none'} | {'yes' if skip else 'no'} | {rec[1]:.3f} | {rec[3]:.3f} | {rec[5]:.3f} | "
                         f"{row['reranked_share']:.2f} | {row['rerank_p50_ms']} | {row['rerank_p95_ms']} |")

    # abstention threshold on the best reranker score among the top-20
    best = {r["id"]: max((c["rerank"] or 0) for c in r["fused"]) if r["fused"] else 0.0 for r in rows}
    una = [r for r in rows if not r["answerable"]]
    lines += ["", "| Threshold on best rerank score | False abstention (answerable) | Abstention (unanswerable) |", "|---|---|---|"]
    for th in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2):
        fa = sum(best[r["id"]] < th for r in ans) / len(ans)
        ab = sum(best[r["id"]] < th for r in una) / len(una)
        out["threshold"].append({"threshold": th, "false_abstention": round(fa, 4), "abstention_unanswerable": round(ab, 4)})
        lines.append(f"| {th} | {fa:.3f} | {ab:.3f} |")
    lines += ["", f"Best-score median: answerable {statistics.median(best[r['id']] for r in ans):.3f}, "
                  f"unanswerable {statistics.median(best[r['id']] for r in una):.3f}"]
    (RESULTS / "rerank_simulation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS / "rerank_simulation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
