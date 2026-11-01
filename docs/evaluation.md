# Evaluation

This document reports the initial evaluation of Al-Murshid: the retrieval ablation, the rerank study, the generation metrics, and the measured latency and cost. The dataset is `data/eval/qa.jsonl` (217 items; 139 Arabic, 57 French, 21 English).

## 1. Retrieval ablation

Keyword (BM25), dense (multilingual-e5-large), hybrid (RRF of the two), and hybrid + cross-encoder reranking (bge-reranker-v2-m3), all evaluated on the answerable items. Relevance is at article level (any chunk covering a gold article counts).

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Search p50 (ms) | Search p95 (ms) |
|---|---|---|---|---|---|---|---|
| bm25 | 0.557 | 0.734 | 0.823 | 0.882 | 0.665 | 1.2 | 2.1 |
| dense | 0.709 | 0.852 | 0.892 | 0.951 | 0.789 | 231.9 | 279.4 |
| hybrid | 0.660 | 0.833 | 0.911 | 0.951 | 0.765 | 227.9 | 274.4 |
| hybrid_rerank | 0.842 | 0.956 | 0.975 | 0.980 | 0.900 | 5788.1 | 7878.3 |

Recall@5 by language:

| Mode | ar | en | fr |
|---|---|---|---|
| bm25 | 0.821 (n=134) | 0.588 (n=17) | 0.904 (n=52) |
| dense | 0.933 (n=134) | 0.588 (n=17) | 0.885 (n=52) |
| hybrid | 0.925 (n=134) | 0.706 (n=17) | 0.942 (n=52) |
| hybrid_rerank | 0.978 (n=134) | 0.882 (n=17) | 1.000 (n=52) |

## 2. Generation metrics

Faithfulness is the share of answer sentences whose cited passages support them, judged by the multilingual cross-encoder used as an entailment proxy (score >= 0.5), combined with deterministic checks that numbers and article numbers in the answer occur in the cited passages. Correctness requires every key fact of the reference answer to appear. Citation precision is the share of citations pointing to a gold article.

Mode `hybrid_rerank`, top_k=5, rerank_k=20, LLM `qwen2.5-3b-instruct-q4_k_m.gguf`, embedder `multilingual-e5-large`, reranker `bge-reranker-v2-m3`.

| Metric | Value |
|---|---|
| Answer correctness | 0.6436 |
| Faithfulness (sentence support) | 0.7887 |
| Citation precision | 0.7587 |
| Answers citing a gold article | 0.8901 |
| Answers with unverified article numbers | 1 |
| Answers with unsupported numbers | 4 |
| Abstention on unanswerable | 0.9286 |
| False abstention on answerable | 0.1034 |
| Forbidden claims | 0 |

Source: `eval/results/generation_baseline_summary.json`.

## 3. Latency and cost

Measured per request on the same run as Section 2 (CPU inference, llama.cpp, local Qwen2.5-3B). Cost assumes the configured provider pricing in `backend/app/core/costs.py`.

| Metric | Value |
|---|---|
| End-to-end latency p50 | 19791.4 ms |
| End-to-end latency p95 | 33235.9 ms |
| End-to-end latency mean | 28060.2 ms |
| Mean input tokens | 1367.2 |
| Mean output tokens | 38.7 |
| Mean cost per query | $0.001026 |
| Cost per 1,000 queries | $1.03 |
| Evaluated queries (n) | 217 |

Source: `eval/results/generation_baseline_summary.json`.

## 4. Threats to validity

- Questions were written from the indexed texts, which makes retrieval easier than real user queries; the plain-language and cross-lingual subsets partially offset this.
- Faithfulness uses a cross-encoder as an entailment proxy, not a dedicated NLI model or human judge.
- **The 100 evaluation items have not been reviewed by a lawyer.** They were checked automatically: every gold reference resolves to an indexed article and every evidence quote occurs verbatim in it. Human legal review is stated as future work.
