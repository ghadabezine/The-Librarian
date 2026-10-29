"""Retrieval evaluation and the four-way ablation.

For every answerable question with a gold article, each retrieval mode
returns its top-10 chunks; we report Recall@k (is a gold passage in the top
k?), MRR@10, gold-article coverage@5 for multi-gold questions, and search
latency (p50/p95, warm, single process, CPU).

    python eval/retrieval_eval.py                       # all four modes
    python eval/retrieval_eval.py --modes hybrid_rerank --tag rerank10 --rerank-k 10
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from common import RESULTS, history_predicate, is_gold, latency_summary, load_items

KS = (1, 3, 5, 10)


def evaluate(retriever, items, mode: str) -> dict:
    rows, lat = [], []
    retriever.retrieve("warm up query الزواج", mode=mode, top_k=10)   # load models before timing
    for it in items:
        t = time.perf_counter()
        res = retriever.retrieve(it["q"], mode=mode, top_k=10, predicate=history_predicate(it["q"]))
        lat.append((time.perf_counter() - t) * 1000)
        flags = [is_gold(h.chunk, it["gold"]) for h in res.hits]
        first = next((i for i, f in enumerate(flags) if f), None)
        found_articles = {f"{h.chunk['doc_id']}:{h.chunk.get('article_label')}" for h in res.hits[:5]}
        rows.append({
            "id": it["id"], "lang": it["lang"], "cat": it["cat"],
            "rank": None if first is None else first + 1,
            "coverage@5": len(found_articles & set(it["gold"])) / len(it["gold"]),
            "retrieved": [h.chunk["chunk_id"] for h in res.hits],
            "latency_ms": lat[-1], "reranked": res.reranked,
        })

    def summarise(sub):
        n = len(sub)
        out = {f"recall@{k}": round(sum(r["rank"] is not None and r["rank"] <= k for r in sub) / n, 4) for k in KS}
        out["mrr@10"] = round(sum(1 / r["rank"] for r in sub if r["rank"]) / n, 4)
        multi = [r for r in sub if r["coverage@5"] is not None]
        out["coverage@5"] = round(sum(r["coverage@5"] for r in multi) / len(multi), 4)
        out["n"] = n
        return out

    by_lang, by_cat = defaultdict(list), defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
        by_cat[r["cat"]].append(r)
    return {
        "mode": mode, "overall": summarise(rows),
        "by_language": {k: summarise(v) for k, v in sorted(by_lang.items())},
        "by_category": {k: summarise(v) for k, v in sorted(by_cat.items())},
        "latency": latency_summary(lat),
        "rerank_skipped": sum(not r["reranked"] for r in rows) if mode == "hybrid_rerank" else None,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid", "hybrid_rerank"])
    ap.add_argument("--tag", default="ablation")
    ap.add_argument("--rerank-k", type=int, default=None)
    ap.add_argument("--candidate-k", type=int, default=None)
    ap.add_argument("--skip-rerank-when-confident", action="store_true")
    args = ap.parse_args()

    from app.core.config import Settings
    from app.retrieval.retriever import Retriever

    overrides = {}
    if args.rerank_k:
        overrides["rerank_k"] = args.rerank_k
    if args.candidate_k:
        overrides["candidate_k"] = args.candidate_k
    if args.skip_rerank_when_confident:
        overrides["skip_rerank_when_confident"] = True
    settings = Settings(**overrides)
    retriever = Retriever(settings)
    items = [i for i in load_items() if i["answerable"] and i["gold"]]
    print(f"{len(items)} answerable questions with gold passages; settings {overrides or 'default'}")

    results = {"settings": {"candidate_k": settings.candidate_k, "rerank_k": settings.rerank_k, "rrf_k": settings.rrf_k,
                            "embedding_model": settings.embedding_model, "reranker_model": settings.reranker_model,
                            "skip_rerank_when_confident": settings.skip_rerank_when_confident,
                            "chunk_strategy": settings.chunk_strategy},
               "n_questions": len(items), "modes": {}}
    for mode in args.modes:
        r = evaluate(retriever, items, mode)
        results["modes"][mode] = r
        o, l = r["overall"], r["latency"]
        print(f"{mode:14s} R@1={o['recall@1']:.3f} R@3={o['recall@3']:.3f} R@5={o['recall@5']:.3f} "
              f"R@10={o['recall@10']:.3f} MRR={o['mrr@10']:.3f} p50={l['p50_ms']}ms p95={l['p95_ms']}ms")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"retrieval_{args.tag}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Search p50 (ms) | Search p95 (ms) |",
             "|---|---|---|---|---|---|---|---|"]
    for mode, r in results["modes"].items():
        o, l = r["overall"], r["latency"]
        lines.append(f"| {mode} | {o['recall@1']:.3f} | {o['recall@3']:.3f} | {o['recall@5']:.3f} | "
                     f"{o['recall@10']:.3f} | {o['mrr@10']:.3f} | {l['p50_ms']} | {l['p95_ms']} |")
    lines += ["", "Recall@5 by language:", "", "| Mode | " + " | ".join(results["modes"][args.modes[0]]["by_language"]) + " |",
              "|---|" + "---|" * len(results["modes"][args.modes[0]]["by_language"])]
    for mode, r in results["modes"].items():
        lines.append(f"| {mode} | " + " | ".join(f"{v['recall@5']:.3f} (n={v['n']})" for v in r["by_language"].values()) + " |")
    (RESULTS / f"retrieval_{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
