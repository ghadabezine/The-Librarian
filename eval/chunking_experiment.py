"""Compare chunking strategies on retrieval quality (same questions, same models).

Strategies: article-aware chunking at two sizes vs. structure-blind sliding
windows at two sizes. Each strategy is indexed in memory (embeddings reuse the
on-disk embedding cache when texts repeat) and evaluated with BM25, dense and
hybrid retrieval (no reranker, to isolate the effect of chunking).

    python eval/chunking_experiment.py
"""
from __future__ import annotations

import json
from pathlib import Path

from common import RESULTS, history_predicate, is_gold, load_items

STRATEGIES = [
    ("article", 1200, 150),
    ("article", 600, 100),
    ("fixed", 800, 150),
    ("fixed", 1500, 250),
]


def main() -> None:
    from app.core.config import Settings
    from app.ingestion.pipeline import _embed_with_cache, build_chunks
    from app.retrieval.embedder import get_embedder
    from app.retrieval.retriever import Retriever
    from app.retrieval.vector_store import VectorIndex

    base = Settings()
    embedder = get_embedder(str(base.models_dir / base.embedding_model), base.device)
    items = [i for i in load_items() if i["answerable"] and i["gold"]]
    results = []
    for strategy, size, overlap in STRATEGIES:
        settings = Settings(chunk_strategy=strategy, chunk_max_chars=size, chunk_overlap_chars=overlap)
        chunks, _ = build_chunks(settings)
        # Embeddings go through a separate cache file so the main index cache is untouched.
        vecs, reused = _embed_with_cache(embedder, [c["embed_text"] for c in chunks], RESULTS / "chunking_cache")
        vindex = VectorIndex(vecs, [c["chunk_id"] for c in chunks], embedder.name)
        retriever = Retriever(settings, chunks=chunks, vector_index=vindex, embedder=embedder)
        row = {"strategy": strategy, "max_chars": size, "overlap": overlap, "chunks": len(chunks),
               "mean_chunk_chars": round(sum(len(c["text"]) for c in chunks) / len(chunks)),
               "embeddings_reused": reused}
        for mode in ("bm25", "dense", "hybrid"):
            hits1 = hits5 = 0
            for it in items:
                res = retriever.retrieve(it["q"], mode=mode, top_k=5, predicate=history_predicate(it["q"]))
                flags = [is_gold(h.chunk, it["gold"]) for h in res.hits]
                hits1 += bool(flags[:1] and flags[0])
                hits5 += any(flags)
            row[f"{mode}_recall@1"] = round(hits1 / len(items), 4)
            row[f"{mode}_recall@5"] = round(hits5 / len(items), 4)
        # context cost: characters sent to the LLM for top-5
        row["context_chars_top5"] = row["mean_chunk_chars"] * 5
        print(row, flush=True)
        results.append(row)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "chunking_experiment.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["| Strategy | Max chars | Overlap | Chunks | Mean chars | BM25 R@5 | Dense R@1 | Dense R@5 | Hybrid R@1 | Hybrid R@5 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['strategy']} | {r['max_chars']} | {r['overlap']} | {r['chunks']} | {r['mean_chunk_chars']} | "
                     f"{r['bm25_recall@5']:.3f} | {r['dense_recall@1']:.3f} | {r['dense_recall@5']:.3f} | "
                     f"{r['hybrid_recall@1']:.3f} | {r['hybrid_recall@5']:.3f} |")
    (RESULTS / "chunking_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
