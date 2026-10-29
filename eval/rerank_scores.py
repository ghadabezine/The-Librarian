"""Record the fused candidate list and reranker scores for every question.

Because the cross-encoder scores each (question, passage) pair independently,
reranking only the first k fused candidates gives exactly the scores of those k
pairs. One run with rerank_k=20 therefore lets us (a) calibrate the abstention
threshold and (b) evaluate smaller rerank depths offline without re-running
the model.

    python eval/rerank_scores.py   ->  eval/results/rerank_scores.jsonl
"""
from __future__ import annotations

import json
import time

from common import RESULTS, history_predicate, is_gold, load_items


def main() -> None:
    from app.core.config import Settings
    from app.retrieval.retriever import Retriever

    settings = Settings(rerank_k=20, top_k=20)
    retriever = Retriever(settings)
    out = RESULTS / "rerank_scores.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    with open(out, "a", encoding="utf-8") as f:
        for it in load_items():
            if it["id"] in done:
                continue
            pred = history_predicate(it["q"])
            fused = retriever.retrieve(it["q"], mode="hybrid", top_k=20, predicate=pred)
            t = time.perf_counter()
            reranked = retriever.retrieve(it["q"], mode="hybrid_rerank", top_k=20, predicate=pred)
            scores = {h.chunk["chunk_id"]: h.rerank_score for h in reranked.hits}
            f.write(json.dumps({
                "id": it["id"], "answerable": it["answerable"], "cat": it["cat"], "lang": it["lang"],
                "fused": [{"chunk_id": h.chunk["chunk_id"], "gold": is_gold(h.chunk, it["gold"]),
                           "rerank": scores.get(h.chunk["chunk_id"]), "bm25_rank": h.bm25_rank,
                           "dense_rank": h.dense_rank, "via_article_ref": h.via_article_ref}
                          for h in fused.hits],
                "rerank_ms": reranked.timings_ms.get("rerank"),
                "embed_ms": reranked.timings_ms.get("embed"),
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(it["id"], round((time.perf_counter() - t), 1), "s", flush=True)


if __name__ == "__main__":
    main()
