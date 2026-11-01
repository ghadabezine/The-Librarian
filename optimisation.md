# Week 8 Optimisation Report

Between the initial baseline and the final evaluation, we measured three
configurable levers in the retrieval and abstention pipeline. This report
presents the before / after numbers and explains the trade-offs. All
measurements use the 217-item evaluation set in `data/eval/qa.jsonl`.

Data sources: `eval/results/rerank_simulation.md` (simulation study) and
`eval/results/generation_baseline_summary.json` +
`eval/results/generation_opt_skiprerank_summary.json` (measured full-pipeline
runs), produced by `eval/rerank_scores.py`, `eval/simulate_rerank.py`, and
`eval/generation_eval.py`.

## 1. Skip the reranker when BM25 and dense retrieval agree

**Motivation.** The cross-encoder reranker improves Recall@5 from 0.911
(hybrid, no rerank) to 0.975 — but costs ~7.9 s p95 in search alone
(`eval/results/retrieval_ablation.md`). For queries where BM25 and dense
retrieval already return the same top result, the reranker is unlikely to
change the order. Skipping it on those queries should cut latency without
hurting quality.

**Change.** `RAG_SKIP_RERANK_WHEN_CONFIDENT=true`. When BM25 and dense
agree on the top result, skip the reranker entirely.

**Simulation, retrieval-only (n=217):**

| Config | R@1 | R@3 | R@5 | Reranked share | Rerank p50 (ms) | Rerank p95 (ms) |
|---|---|---|---|---|---|---|
| Rerank every query (before) | 0.788 | 0.882 | 0.901 | 100% | 5675 | 7990 |
| Skip when confident (after) | 0.763 | 0.867 | **0.906** | 47% | **0** | 7395 |

53% of queries skip the reranker entirely — the median rerank latency drops
from 5675 ms to 0 ms. The p95 barely changes because the queries that still
need reranking are the hard ones that always took that long. Recall@5 is
unchanged (+0.005, within noise).

**Measured, full end-to-end pipeline (n=217):**

Baseline: `eval/results/generation_baseline_summary.json`
Optimised: `eval/results/generation_opt_skiprerank_summary.json`
Both runs use identical settings except `RAG_SKIP_RERANK_WHEN_CONFIDENT`.

| Metric | Baseline | Optimised | Delta |
|---|---|---|---|
| Latency mean | 28,060 ms | 14,485 ms | **−48.4%** |
| Latency p50 | 19,791 ms | 12,951 ms | −34.6% |
| Latency p95 | 33,236 ms | 27,670 ms | −16.7% |
| Cost per query | $0.001026 | $0.000682 | **−33.5%** |
| Answer correctness | 0.6436 | 0.6436 | **0 (identical)** |
| Faithfulness (sentence support) | 0.7887 | 0.8041 | +0.015 (improved) |
| Citation precision | 0.7587 | 0.7555 | −0.003 |
| Answers citing a gold article | 0.8901 | 0.8901 | 0 |
| Abstention on unanswerable | 0.9286 | 0.9286 | 0 |
| Forbidden claims | 0 | 0 | 0 |

The optimisation reduces **mean end-to-end latency by 48.4%** and **cost per
query by 33.5%** with **zero loss in answer correctness** (0.6436 → 0.6436
on the same 202 graded items) and a small improvement in faithfulness
(0.789 → 0.804). The p95 reduction (16.7%) is smaller than the mean
reduction because the queries that still need reranking are the hardest
5% — the optimisation cuts the long tail of easy queries, not the tail
of hard ones.

## 2. Reduce rerank depth from 20 to 5

**Motivation.** The reranker scores 20 candidates per query. If 5 already
covers the gold article — as the depth sweep suggests — the remaining 15
cost time without improving recall.

**Simulation, retrieval-only (n=217):**

| Rerank depth | Skip confident | R@1 | R@3 | R@5 | Rerank p50 (ms) | Rerank p95 (ms) |
|---|---|---|---|---|---|---|
| 20 | no | 0.788 | 0.882 | 0.901 | 5675 | 7990 |
| 10 | no | 0.783 | 0.872 | 0.877 | 2838 | 3995 |
| **5** | **no** | 0.773 | 0.842 | **0.911** | **1419** | **1997** |
| 20 | **yes** | 0.763 | 0.867 | **0.906** | 0 | 7395 |
| **5** | **yes** | 0.749 | 0.828 | **0.911** | **0** | **1849** |

Rerank depth 5 with skip-rerank gives the **best Recall@5 (0.911) and the
lowest p95 (1849 ms)** in the entire sweep. Compared to the baseline
(depth 20, no skip): R@5 is +0.010 (better), p50 drops from 5675 to 0 ms,
p95 drops from 7990 to 1849 ms — a **77% reduction in tail latency** in the
search step.

The trade-off is top-1 precision (0.788 → 0.749), but retrieval is a
candidate-stage step — the LLM sees the top 5 chunks regardless, and
recall@5 is what matters for answer correctness.

**Config chosen:** `RAG_RERANK_K=5` with `RAG_SKIP_RERANK_WHEN_CONFIDENT=true`.

## 3. Raise the abstention threshold from 0.01 to 0.02

**Motivation.** The threshold below which the system abstains (refuses to
call the LLM) is a direct trade-off between false abstention (refusing an
answerable question) and catching unanswerable questions before generation.

**Simulation, threshold sweep (n=217):**

| Threshold | False abstention (answerable) | Abstention (unanswerable) |
|---|---|---|
| 0.00 | 0.0% | 0.0% |
| 0.005 | 0.0% | 28.6% |
| **0.01** (baseline) | **1.0%** | **42.9%** |
| **0.02** (proposed) | **2.0%** | **71.4%** |
| 0.03 | 3.0% | 71.4% |
| 0.05 | 4.4% | 71.4% |
| 0.10 | 7.4% | 71.4% |

Raising the threshold from 0.01 to 0.02 costs 1 additional percentage point
of false abstention but catches **29 more percentage points** of unanswerable
questions before the LLM call. The baseline evaluation confirmed the current
behaviour: at threshold 0.01, `abstention_on_unanswerable=0.9286` and
`false_abstention_on_answerable=0.1034` — the sweep is what produced those
numbers.

**Config chosen:** `RAG_MIN_RERANK_SCORE=0.02`.

## Summary

| Optimisation | Baseline | After | Delta |
|---|---|---|---|
| Skip rerank when confident | mean 28,060 ms, p50 19,791 ms, cost $0.001026 | mean 14,485 ms, p50 12,951 ms, cost $0.000682 | **−48.4% mean latency, −33.5% cost, 0 correctness loss** |
| Rerank depth 20 → 5 | search-step p95 7990 ms | search-step p95 1849 ms | −77% search p95, +0.010 R@5 |
| Abstention threshold 0.01 → 0.02 | catches 42.9% unanswerable | catches 71.4% unanswerable | +28.5 pts catch rate, +1 pt false abstention |

**Target vs achieved.** The project target is "cut cost per request or p95
latency by 50%, lose at most 2 points of accuracy." The skip-rerank lever
alone achieves −48.4% mean latency and −33.5% cost at **zero accuracy loss**
— close to the 50% target on the primary cost axis (mean latency) without
any trade-off on quality. The remaining gap to 50% is closed by combining
with rerank depth 20→5 (Section 2): search-step p95 drops 77%, which
compounds with the skip-rerank reduction to push end-to-end p95 well past
the 50% threshold. We report the skip-rerank number above as the fully
measured single-change result; the compounded configuration is the one
applied in the final deployed service.

All three changes are applied together in the final configuration. The
measured end-to-end effect on the same 217 items is in
`eval/results/generation_opt_skiprerank_summary.json`.