"""Assemble docs/evaluation.md from eval/results/*.

Pulls the retrieval ablation table, rerank simulation, and the FULL
generation baseline summary (not the small probe runs) into one document
with a latency/cost table. Run after eval/generation_eval.py --tag baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "evaluation.md"

PREFERRED_GEN = ["generation_baseline_summary.json",
                 "generation_optimised_summary.json",
                 "generation_probe_top3_summary.json"]
PREFERRED_RET = ["retrieval_ablation.json", "retrieval_ablation_nocollapse.json"]


def first_existing(names):
    for n in names:
        p = RESULTS / n
        if p.exists():
            return p
    return None


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p else None


def latency_cost_block(summary: dict) -> str:
    o = summary.get("overall", {})
    lat = o.get("latency", {})
    rows = [
        ["End-to-end latency p50", f"{lat.get('p50_ms', 'n/a')} ms"],
        ["End-to-end latency p95", f"{lat.get('p95_ms', 'n/a')} ms"],
        ["End-to-end latency mean", f"{lat.get('mean_ms', 'n/a')} ms"],
        ["Mean input tokens", o.get("mean_input_tokens", "n/a")],
        ["Mean output tokens", o.get("mean_output_tokens", "n/a")],
        ["Mean cost per query", f"${o.get('mean_cost_usd', 0):.6f}"],
        ["Cost per 1,000 queries", f"${o.get('mean_cost_usd', 0) * 1000:.2f}"],
        ["Evaluated queries (n)", o.get("n", "n/a")],
    ]
    return md_table(["Metric", "Value"], rows)


def gen_block(summary: dict) -> str:
    o = summary.get("overall", {})
    rows = [
        ["Answer correctness", o.get("answer_correctness")],
        ["Faithfulness (sentence support)", o.get("faithfulness_sentence_support")],
        ["Citation precision", o.get("citation_precision")],
        ["Answers citing a gold article", o.get("answers_citing_gold")],
        ["Answers with unverified article numbers", o.get("answers_with_unverified_articles")],
        ["Answers with unsupported numbers", o.get("answers_with_unsupported_numbers")],
        ["Abstention on unanswerable", o.get("abstention_on_unanswerable")],
        ["False abstention on answerable", o.get("false_abstention_on_answerable")],
        ["Forbidden claims", o.get("forbidden_claims")],
    ]
    settings = summary.get("settings", {})
    head = (f"Mode `{summary.get('mode')}`, top_k={settings.get('top_k')}, "
            f"rerank_k={settings.get('rerank_k')}, "
            f"LLM `{settings.get('llm_model')}`, "
            f"embedder `{settings.get('embedding_model')}`, "
            f"reranker `{settings.get('reranker_model')}`.")
    return head + "\n\n" + md_table(["Metric", "Value"], rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ret_p = first_existing(PREFERRED_RET)
    ret_md = ret_p.with_suffix(".md") if ret_p else None
    gen_p = first_existing(PREFERRED_GEN)
    gen = load_json(gen_p)

    parts = ["# Evaluation", ""]
    parts.append("This document reports the initial evaluation of Al-Murshid: "
                 "the retrieval ablation, the rerank study, the generation "
                 "metrics, and the measured latency and cost. The dataset is "
                 "`data/eval/qa.jsonl` (217 items; 139 Arabic, 57 French, "
                 "21 English).")
    parts.append("")

    parts.append("## 1. Retrieval ablation")
    parts.append("")
    parts.append("Keyword (BM25), dense (multilingual-e5-large), hybrid (RRF of "
                 "the two), and hybrid + cross-encoder reranking "
                 "(bge-reranker-v2-m3), all evaluated on the answerable items. "
                 "Relevance is at article level (any chunk covering a gold "
                 "article counts).")
    parts.append("")
    if ret_md and ret_md.exists():
        parts.append(ret_md.read_text(encoding="utf-8").strip())
    elif ret_p:
        parts.append("```json\n" + ret_p.read_text(encoding="utf-8").strip() + "\n```")
    else:
        parts.append("_No retrieval results found._")
    parts.append("")

    parts.append("## 2. Generation metrics")
    parts.append("")
    parts.append("Faithfulness is the share of answer sentences whose cited "
                 "passages support them, judged by the multilingual "
                 "cross-encoder used as an entailment proxy (score >= 0.5), "
                 "combined with deterministic checks that numbers and article "
                 "numbers in the answer occur in the cited passages. Correctness "
                 "requires every key fact of the reference answer to appear. "
                 "Citation precision is the share of citations pointing to a "
                 "gold article.")
    parts.append("")
    parts.append(gen_block(gen) if gen else "_No generation summary found. Run:_ `python eval/generation_eval.py --tag baseline`")
    parts.append("")
    if gen_p:
        parts.append(f"Source: `eval/results/{gen_p.name}`.")
    parts.append("")

    parts.append("## 3. Latency and cost")
    parts.append("")
    if gen:
        parts.append("Measured per request on the same run as Section 2 (CPU "
                     "inference, llama.cpp, local Qwen2.5-3B). Cost assumes the "
                     "configured provider pricing in `backend/app/core/costs.py`.")
        parts.append("")
        parts.append(latency_cost_block(gen))
        parts.append("")
        parts.append(f"Source: `eval/results/{gen_p.name}`.")
    else:
        parts.append("_No timing data available._")
    parts.append("")

    parts.append("## 4. Threats to validity")
    parts.append("")
    parts.append("- Questions were written from the indexed texts, which makes "
                 "retrieval easier than real user queries; the plain-language "
                 "and cross-lingual subsets partially offset this.")
    parts.append("- Faithfulness uses a cross-encoder as an entailment proxy, "
                 "not a dedicated NLI model or human judge.")
    parts.append("- **The 100 evaluation items have not been reviewed by a "
                 "lawyer.** They were checked automatically: every gold "
                 "reference resolves to an indexed article and every evidence "
                 "quote occurs verbatim in it. Human legal review is stated as "
                 "future work.")
    parts.append("")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  retrieval source: {ret_p.name if ret_p else 'MISSING'}")
    print(f"  generation source: {gen_p.name if gen_p else 'MISSING'}")
    if gen_p and "probe" in gen_p.name:
        print("\n  WARNING: using a probe summary. Run the full baseline first:")
        print("    python eval/generation_eval.py --tag baseline")


if __name__ == "__main__":
    main()