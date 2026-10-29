"""End-to-end generation evaluation over data/eval/qa.jsonl.

Runs the full pipeline (retrieval + LLM + verification) on every question and
measures, separately from retrieval:

* answer correctness  – every key-fact group of the reference must appear in
  the answer (normalised string match against listed variants)
* faithfulness        – share of answer sentences whose cited passage
  supports them, judged by the multilingual cross-encoder (bge-reranker-v2-m3)
  used as an entailment proxy (score >= 0.5), plus the deterministic checks:
  numbers and article numbers in the answer must appear in the cited passages
* citation accuracy   – share of citations that point to a gold article, and
  share of answers with at least one gold citation
* abstention          – unanswerable questions answered with an abstention;
  answerable questions wrongly refused
* forbidden strings   – hallucination-bait answers containing banned claims
* latency / tokens / cost per request

    python eval/generation_eval.py --tag baseline
    python eval/generation_eval.py --tag optimised --top-k 3 --max-context-chars 3000 ...
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from common import RESULTS, is_gold, latency_summary, load_items

SUPPORT_THRESHOLD = 0.5


def fact_hits(answer: str, facts: list[list[str]]) -> list[bool]:
    from app.ingestion.normalize import search_form
    a = " " + search_form(answer) + " "
    return [any(search_form(v) in a for v in group) for group in facts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--mode", default="hybrid_rerank")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--rerank-k", type=int)
    ap.add_argument("--max-context-chars", type=int)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--llm-model")
    ap.add_argument("--provider")
    ap.add_argument("--min-rerank-score", type=float)
    ap.add_argument("--skip-rerank-when-confident", action="store_true")
    ap.add_argument("--prompt-cache", action="store_true", help="reuse the KV cache of the fixed prompt prefix")
    ap.add_argument("--query-cache", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--fresh", action="store_true", help="discard earlier rows for this tag")
    args = ap.parse_args()

    from app.core.config import Settings
    from app.generation.generator import RAGService
    from app.retrieval.reranker import get_reranker

    overrides = {k: v for k, v in {
        "top_k": args.top_k, "rerank_k": args.rerank_k, "max_context_chars": args.max_context_chars,
        "llm_max_tokens": args.max_tokens, "llm_model": args.llm_model, "llm_provider": args.provider,
        "min_rerank_score": args.min_rerank_score, "query_cache_size": args.query_cache,
        "skip_rerank_when_confident": args.skip_rerank_when_confident or None,
        "llm_prompt_cache": args.prompt_cache or None,
    }.items() if v is not None}
    settings = Settings(**overrides)
    svc = RAGService(settings)
    items = load_items()
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.limit:
        items = items[: args.limit]

    svc.answer("ما هو السن الأدنى للزواج؟", mode=args.mode)   # warm-up: load all models
    judge = get_reranker(str(settings.models_dir / settings.reranker_model), settings.device)

    out_path = RESULTS / f"generation_{args.tag}.jsonl"
    rows = []
    # Resumable: rows already written for this tag are kept and skipped.
    if out_path.exists() and not args.fresh:
        with open(out_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    done = {r["id"] for r in rows}
    with open(out_path, "a" if rows else "w", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            t = time.perf_counter()
            ans = svc.answer(it["q"], mode=args.mode)
            wall = (time.perf_counter() - t) * 1000
            from app.generation.verify import CITATION, split_sentences
            cited = ans.citations
            row = {
                "id": it["id"], "lang": it["lang"], "cat": it["cat"], "answerable": it["answerable"],
                "question": it["q"], "answer": ans.answer, "status": ans.status,
                "abstain_reason": ans.abstain_reason, "latency_ms": wall,
                "timings": ans.timings_ms, "usage": ans.usage,
                "cited": [c["chunk_id"] for c in cited],
                "retrieved": [s["chunk_id"] for s in ans.sources],
                "verification": {k: v for k, v in ans.verification.items() if k != "uncited_sentences"},
            }
            if ans.status == "answered":
                by_id = {c["id"]: c for c in ans.sources}
                gold_flags = [is_gold(c, it["gold"]) for c in cited]
                row["citation_precision"] = sum(gold_flags) / len(gold_flags) if gold_flags else 0.0
                row["gold_cited"] = any(gold_flags)
                # Faithfulness: score each cited sentence against its cited passages.
                scores = []
                for sent in split_sentences(ans.answer):
                    ids = [f"S{m}" for m in CITATION.findall(sent)]
                    passages = [by_id[i]["text"] for i in ids if i in by_id]
                    if not passages:
                        continue
                    claim = CITATION.sub("", sent).strip()
                    scores.append(float(max(judge.score(claim, passages))))
                row["support_scores"] = scores
                row["faithful_sentences"] = sum(s >= SUPPORT_THRESHOLD for s in scores)
                row["cited_sentences"] = len(scores)
                if it["facts"]:
                    hits = fact_hits(ans.answer, it["facts"])
                    row["fact_hits"] = hits
                    row["correct"] = all(hits)
                row["forbidden_found"] = [s for s in it.get("forbid", []) if s in ans.answer]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(f"[{n}/{len(items)}] {it['id']} {ans.status:9s} {wall / 1000:6.1f}s "
                  f"{'correct' if row.get('correct') else ('-' if 'correct' not in row else 'WRONG')}", flush=True)

    wanted = {i["id"] for i in items}
    rows = [r for r in rows if r["id"] in wanted]
    summary = summarise(rows, settings, args)
    (RESULTS / f"generation_{args.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                                   encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=1))


def summarise(rows: list[dict], settings, args) -> dict:
    def block(sub):
        answerable = [r for r in sub if r["answerable"]]
        unanswerable = [r for r in sub if not r["answerable"]]
        graded = [r for r in answerable if "correct" in r or (r["status"] == "abstained")]
        # Correctness on answerable questions with key facts: an abstention counts as wrong.
        with_facts = [r for r in answerable if r.get("fact_hits") is not None or r["status"] == "abstained"]
        answered = [r for r in sub if r["status"] == "answered"]
        cited_sents = sum(r.get("cited_sentences", 0) for r in answered)
        faithful = sum(r.get("faithful_sentences", 0) for r in answered)
        prec = [r["citation_precision"] for r in answered if r["answerable"] and "citation_precision" in r]
        lat = [r["latency_ms"] for r in sub]
        tokens_in = [r["usage"].get("input_tokens", 0) for r in sub]
        tokens_out = [r["usage"].get("output_tokens", 0) for r in sub]
        cost = [r["usage"].get("cost_usd", 0.0) for r in sub]
        return {
            "n": len(sub),
            "answer_correctness": round(sum(bool(r.get("correct")) for r in with_facts) / len(with_facts), 4) if with_facts else None,
            "n_graded_correctness": len(with_facts),
            "faithfulness_sentence_support": round(faithful / cited_sents, 4) if cited_sents else None,
            "answers_with_unsupported_numbers": sum(bool(r["verification"].get("unsupported_numbers")) for r in answered),
            "answers_with_unverified_articles": sum(bool(r["verification"].get("unverified_articles")) for r in answered),
            "citation_precision": round(sum(prec) / len(prec), 4) if prec else None,
            "answers_citing_gold": round(sum(r.get("gold_cited", False) for r in answered if r["answerable"]) /
                                         max(1, sum(1 for r in answered if r["answerable"])), 4),
            "abstention_on_unanswerable": round(sum(r["status"] == "abstained" for r in unanswerable) / len(unanswerable), 4) if unanswerable else None,
            "false_abstention_on_answerable": round(sum(r["status"] == "abstained" for r in answerable) / len(answerable), 4) if answerable else None,
            "forbidden_claims": sum(bool(r.get("forbidden_found")) for r in sub),
            "latency": latency_summary(lat),
            "mean_input_tokens": round(sum(tokens_in) / len(sub), 1),
            "mean_output_tokens": round(sum(tokens_out) / len(sub), 1),
            "mean_cost_usd": sum(cost) / len(sub),
        }

    by_lang, by_cat = defaultdict(list), defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
        by_cat[r["cat"]].append(r)
    return {
        "tag": args.tag, "mode": args.mode,
        "settings": {k: getattr(settings, k) for k in ("top_k", "rerank_k", "candidate_k", "max_context_chars",
                                                      "llm_provider", "llm_model", "llm_max_tokens", "min_rerank_score",
                                                      "skip_rerank_when_confident", "query_cache_size", "llm_prompt_cache",
                                                      "embedding_model", "reranker_model")},
        "support_threshold": SUPPORT_THRESHOLD,
        "overall": block(rows),
        "by_language": {k: block(v) for k, v in sorted(by_lang.items())},
        "by_category": {k: block(v) for k, v in sorted(by_cat.items())},
    }


if __name__ == "__main__":
    main()
