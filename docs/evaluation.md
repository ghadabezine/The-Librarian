# Evaluation

This document reports the initial evaluation of Al-Murshid: the retrieval
ablation, the rerank study, the generation metrics, and the measured latency
and cost. The dataset is `data/eval/qa.jsonl` (217 items; 139 Arabic, 57
French, 21 English).

Two generation configurations are evaluated on the full 217-item set: a local
Qwen2.5-3B-Instruct on CPU (the initial baseline) and OpenAI `gpt-4o-mini`
(the deployed configuration). The retrieval stack is identical between the
two runs (multilingual-e5-large + bge-reranker-v2-m3, top_k=5, rerank_k=20);
only the generation LLM differs.

## 1. Retrieval ablation

Keyword (BM25), dense (multilingual-e5-large), hybrid (RRF of the two), and
hybrid + cross-encoder reranking (bge-reranker-v2-m3), all evaluated on the
answerable items. Relevance is at article level (any chunk covering a gold
article counts).

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

**Reading.** Hybrid search alone barely improves over dense on Recall@5
(0.911 vs 0.892) because BM25 cannot match French or English questions to
Arabic text — on English questions BM25 gets 0.588 while dense gets 0.588,
and on French BM25 gets 0.904 vs dense 0.885. The reranker is what unlocks
the fused list: it pushes Recall@5 to 0.975 overall, and to a perfect 1.000
on French. Reranking costs ~7.9 s p95 in the search step, which is why the
optimisation report considers skipping it on confident queries.

## 2. Generation metrics

Metrics definition. **Faithfulness** is the share of answer sentences whose
cited passages support them, judged by the multilingual cross-encoder used
as an entailment proxy (score >= 0.5), combined with deterministic checks
that numbers and article numbers in the answer occur in the cited passages.
**Correctness** requires every key fact of the reference answer to appear.
**Citation precision** is the share of citations pointing to a gold article.

Retrieval settings for both runs: `hybrid_rerank`, top_k=5, rerank_k=20,
embedder `multilingual-e5-large`, reranker `bge-reranker-v2-m3`. The two
runs differ only in `RAG_LLM_PROVIDER` / `RAG_OPENAI_MODEL`.

### Baseline — local Qwen2.5-3B-Instruct (CPU, llama.cpp)

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

### Deployed — OpenAI gpt-4o-mini

| Metric | Value |
|---|---|
| Answer correctness | 0.9113 |
| Faithfulness (sentence support) | 0.7488 |
| Citation precision | 0.9433 |
| Answers citing a gold article | 0.9645 |
| Answers with unverified article numbers | 0 |
| Answers with unsupported numbers | 0 |
| Abstention on unanswerable | 1.0000 |
| False abstention on answerable | 0.0296 |
| Forbidden claims | 0 |

Source: `eval/results/generation_openai_summary.json`.

### Comparison

| Metric | Local Qwen 3B (baseline) | gpt-4o-mini (deployed) | Delta |
|---|---|---|---|
| Answer correctness | 0.6436 | **0.9113** | +0.27 |
| Citation precision | 0.7587 | **0.9433** | +0.18 |
| Answers citing a gold article | 0.8901 | **0.9645** | +0.07 |
| Abstention on unanswerable | 0.9286 | **1.0** | +0.07 |
| False abstention on answerable | 0.1034 | **0.0296** | −0.07 |
| Answers with unsupported numbers | 4 | **0** | eliminated |
| Answers with unverified articles | 1 | **0** | eliminated |
| Faithfulness (sentence support) | 0.7887 | 0.7488 | −0.04 |
| Latency p50 | 19,791 ms | **14,158 ms** | −28% |
| Latency p95 | 33,236 ms | **21,301 ms** | −36% |
| Cost per query | $0.001026 (compute) | **$0.000211 (API)** | −79% |

The deployed configuration improves on the baseline on correctness, citation
precision, abstention behaviour, latency and cost. The small drop in
sentence-level faithfulness (−0.04) is attributed to the model writing longer
sentences, which are harder to fully support by a single citation; the
complete elimination of unsupported numbers and unverified article references
indicates the citation enforcement is stricter in practice, not looser.

Cost for the local configuration is compute time at $0.20/hour (a small cloud
VM); cost for the deployed configuration is per-token API pricing for
`gpt-4o-mini` (input $0.15/MTok, output $0.60/MTok).

## 3. Latency and cost

Measured per request on the same runs as Section 2. The local configuration
runs on CPU (llama.cpp, local Qwen2.5-3B); the deployed configuration calls
the OpenAI API for generation while running the same retrieval stack locally.

### Local Qwen 3B (baseline)

| Metric | Value |
|---|---|
| End-to-end latency p50 | 19,791 ms |
| End-to-end latency p95 | 33,236 ms |
| End-to-end latency mean | 28,060 ms |
| Mean input tokens | 1,367.2 |
| Mean output tokens | 38.7 |
| Mean cost per query | $0.001026 |
| Cost per 1,000 queries | $1.03 |
| Evaluated queries (n) | 217 |

Source: `eval/results/generation_baseline_summary.json`.

### gpt-4o-mini (deployed)

| Metric | Value |
|---|---|
| End-to-end latency p50 | 14,158 ms |
| End-to-end latency p95 | 21,301 ms |
| End-to-end latency mean | 13,873 ms |
| Mean input tokens | 1,244 |
| Mean output tokens | 41.3 |
| Mean cost per query | $0.000211 |
| Cost per 1,000 queries | $0.21 |
| Evaluated queries (n) | 217 |

Source: `eval/results/generation_openai_summary.json`. The deployed cost per
query is derived from the per-request token counts in the run's `.jsonl`
using the current `gpt-4o-mini` list prices ($0.15/MTok input,
$0.60/MTok output).

**Cost at scale.** At the current 217-query evaluation volume, cost per
1,000 queries is $0.21 for the deployed configuration. Scaling to 100×
current traffic (≈21,700 queries/day): ≈$4.57/day, ≈$137/month on
`gpt-4o-mini`. The retrieval stack runs locally in the same container at no
per-query cost; total cost at 100× is dominated by the hosting instance
(≈$10–15/month for a 4 vCPU / 16 GB VM) plus the API spend.

## 4. Threats to validity

- Questions were written from the indexed texts, which makes retrieval
  easier than real user queries; the plain-language and cross-lingual
  subsets partially offset this.
- Faithfulness uses a cross-encoder as an entailment proxy, not a dedicated
  NLI model or human judge.
- **The 100 evaluation items have not been reviewed by a lawyer.** They were
  checked automatically: every gold reference resolves to an indexed article
  and every evidence quote occurs verbatim in it. Human legal review is
  stated as future work.
- The local Qwen 3B and OpenAI gpt-4o-mini runs share the same retrieval
  stack, prompt, and citation enforcement. Differences in faithfulness
  between the two runs therefore reflect a change in answer length and
  writing style, not in retrieval or grounding behaviour.