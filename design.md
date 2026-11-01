# Design — Al-Murshid

How Al-Murshid answers Tunisian personal status law questions: corpus and
provenance, ingestion and chunking, retrieval, generation, verification,
and evaluation. Implementation lives in `backend/app/`.

## 1. Corpus and provenance

The corpus is the Personal Status Code (مجلة الأحوال الشخصية / Code du
statut personnel) and 13 related laws and decrees from the Imprimerie
Officielle de la République Tunisienne (IORT). Every document is listed in
`data/sources.yaml` with source URL, edition year, language, `authority`,
and `legal_status`:

- **`consolidated`** — the current text (IORT 2019 Arabic edition, revised
  to 2017; IORT 2024 French edition).
- **`historical`** — an earlier consolidation (CAWTAR copy), kept only to
  answer "what did the text say before the amendment?" questions and
  excluded from default retrieval unless the query signals historical
  intent.

The distinction matters: Tunisian personal status law has been amended
repeatedly (marriage-age, 1981 divorce amendments, 1993 custody, 2017
inheritance). A single undated snapshot would answer some questions with
law that no longer applies. `legal_status` is carried through chunking
into chunk metadata and checked at query time.

The corpus covers 260 pages across 13 documents, in Arabic and French.
Three recent texts exist only in French and are indexed as French-only.

## 2. Ingestion

`backend/app/ingestion/` is a five-stage pipeline:

1. **Load** (`loader.py`). The Arabic 2019 edition is a scanned PDF and is
   OCR'd; per-page text is cached in `data/ocr/<doc_id>/page_NNN.txt`. The
   French 2024 edition is a text PDF and is extracted directly. OCR output
   is stored so the pipeline is reproducible without re-running OCR. 19 of
   131 pages are near-blank (covers, tables of contents); none contain
   articles, and no evaluation row references them.
2. **Normalise** (`normalize.py`). Arabic orthographic normalisation (alef
   variants, taa marbuta, tatweel, diacritics) produces a `search_form`
   used for both indexing and fact matching.
3. **Structure** (`structure.py`). Article boundaries are detected from
   `الفصل N` (Arabic) and `Article N` (French) markers. Article number,
   language, and any amendment note are captured. Cross-language alignment
   maps French to Arabic by article number, so `csp:N` refers to the same
   article in either edition.
4. **Chunk** (`chunker.py`). Article-aware: an article is one chunk when it
   fits, otherwise split at paragraph boundaries. Fixed-window fallback
   covers front matter and tables. Each chunk carries `chunk_id`, `doc_id`,
   `article_label`, `covered_articles` (fixed-window chunks may span
   articles), `legal_status`, and `heading`.
5. **Index** (`scripts/build_index.py`). BM25 statistics and dense
   embeddings are written to `data/index/`.

## 3. Retrieval

`backend/app/retrieval/` exposes four modes via `RAG_RETRIEVAL_MODE`:

| Mode | Backend |
|---|---|
| `bm25` | `bm25.py` — keyword; handles the `article N` pattern directly |
| `dense` | `embedder.py` + `vector_store.py` — multilingual-e5-large |
| `hybrid` | `fusion.py` — RRF over BM25 and dense |
| `hybrid_rerank` | + `reranker.py` — bge-reranker-v2-m3 (default) |

**Query analysis** (`query.py`) runs before retrieval: language detection,
`article N` / `الفصل N` extraction (which short-circuits to a direct lookup),
and FR/EN → AR expansion so a French or English question retrieves the
Arabic consolidated article even when the French edition is not the
governing text.

    question -> query analysis -> BM25 --+
                                       +-> RRF -> reranker -> top-k -> LLM -> verify -> answer
                             dense ----+                                       |
                                                                          abstain if weak

The reranker threshold (`RAG_MIN_RERANK_SCORE`) was calibrated with
`eval/rerank_scores.py`; rerank depth (`RAG_RERANK_K`) with
`eval/simulate_rerank.py`. Both results are in
`eval/results/rerank_simulation.md`.

## 4. Generation and verification

`backend/app/generation/` generates from the top-k retrieved chunks with a
local Qwen2.5-3B-Instruct (Q4_K_M, llama.cpp, CPU). The prompt
(`prompts.py`) requires citation of each legal claim with `[S1]`-style
markers pointing at numbered passages. `RAG_LLM_PROVIDER` can switch to
`anthropic` (hosted) or `extractive` (no LLM, for debugging retrieval).

`verify.py` runs after generation and enforces three things:

- **Citations resolve.** Every `[Sx]` must refer to a passage actually in
  context; unresolved markers are stripped; if none survive, the answer
  is abstained.
- **Cited sentences are supported.** Each sentence with a citation is
  scored against its cited passages by the cross-encoder as an entailment
  proxy (threshold 0.5).
- **Numbers and article numbers appear in cited passages.** Any number or
  article identifier not present in a cited passage triggers
  `unsupported_numbers` or `unverified_articles`.

The frontend renders citations as clickable links to the article text.

## 5. Evaluation design

The evaluation set (`data/eval/qa.jsonl`) contains 217 items: 139 Arabic,
57 French, 21 English; covering factual, article-identification,
cross-reference, ambiguous, unanswerable, hallucination-bait, historical,
and multi-hop questions. Every item carries a gold article list
(`gold: ["csp:5", ...]`), a verbatim evidence quote (`ev`) from a gold
article, and the chunk-level ids of those articles (`gold_chunk_ids`).
Unanswerable items carry `answerable: false` and a `forbid` list of
strings that must not appear in the answer.

Items were authored from the indexed texts and checked automatically by
`scripts/build_eval_set.py`: every gold reference must resolve to an
indexed article, the evidence quote must occur verbatim in a gold article
after normalisation, and every key fact of the reference answer must
appear in the evidence. Human legal review of 100 selected items is
recorded per item under `verification.human_legal_review` (see
`docs/evaluation.md` §4).

Evaluation is split by phase:

- **Retrieval** (`eval/retrieval_eval.py`): Recall@1, Recall@5, MRR@10 on
  the answerable items, across the four modes. Relevance is at article
  level (a retrieved chunk counts if any covered article is gold).
- **Reranking** (`eval/rerank_scores.py`, `eval/simulate_rerank.py`):
  score distribution on gold vs non-gold, sweep over rerank depth.
- **Generation** (`eval/generation_eval.py`): full pipeline per item;
  reports answer correctness, faithfulness, citation precision,
  abstention, forbidden-claim rate, latency, tokens, cost. Resumable per
  `--tag`.

Published results are in `docs/evaluation.md`.