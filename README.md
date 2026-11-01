# Al-Murshid — Tunisian Personal Status Law RAG

**المرشد القانوني** answers questions about Tunisian personal status law
(مجلة الأحوال الشخصية / Code du statut personnel) in Arabic, French and
English. Every answer is grounded in the official texts, with a citation per
legal claim that opens the exact article. When the texts do not support an
answer, it says so.

CS496 AI Engineering — Project 2 "The Librarian", MedTech, Fall 2026.

> Legal information, not legal advice.

## What it does

- **Corpus**: 260 pages of official legislation from the Imprimerie
  Officielle de la République Tunisienne (IORT):
  - the Personal Status Code and 13 related laws and decrees, in Arabic
    (2019 edition, revised to 2017) and in French (2024 edition);
  - 3 recent texts that exist only in the French edition;
  - a CAWTAR copy with earlier wordings, used only for historical
    questions.

  Details: [`design.md`](design.md) §1 and [`data/sources.yaml`](data/sources.yaml).
- **Pipeline**:
  - OCR (Arabic edition) or text extraction (French edition);
  - article-aware chunking with amendment and heading metadata;
  - BM25 plus multilingual dense retrieval (e5-large), fused with RRF;
  - a cross-encoder reranker (bge-reranker-v2-m3);
  - a local LLM (Qwen2.5-3B, llama.cpp, CPU) that must cite `[S1]`-style
    sources;
  - citation, number and article-number checks, and abstention.
- **App**: a FastAPI backend (`/health`, `/chat`, `/search`, `/documents`,
  `/feedback`) and a React + Tailwind chat UI with Arabic right-to-left
  layout, clickable citations, an article panel, legal-status badges and
  feedback.

```
question ─► query analysis ─► BM25 ─┐
            (language, "article N",  ├─► RRF ─► reranker ─► top-5 articles ─► LLM ─► citation checks ─► answer + sources
             FR/EN→AR expansion)     │                       (abstain if weak)      (abstain if uncited)
                                 dense┘
```

## Quick start

Prerequisites: Python 3.11+ (tested with 3.13), Node 20+, about 6 GB of disk
for the models, and a C compiler for `llama-cpp-python`.

**One-command setup** (models, Python and Node dependencies, index):

```bash
make setup
```

**One-command run** (backend on :8000, frontend on :5173):

```bash
make run
```

Open http://localhost:5173 and ask, for example, *ما هو السن القانوني للزواج في تونس؟*

### With Docker

```bash
cp .env.example .env
docker compose up --build
```

Open http://localhost:8080. On first start the backend downloads the models
into `./models` and builds the index into `./data/index`; both persist in
volumes.

### Manual steps

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements-dev.txt
python scripts/fetch_models.py           # embedder, reranker, Qwen2.5-3B GGUF (~5 GB)
python scripts/build_index.py            # chunks + embeddings -> data/index/
uvicorn app.main:app --app-dir backend --port 8000
cd frontend && npm ci && npm run dev     # http://localhost:5173
```

## Configuration

All settings are environment variables with the `RAG_` prefix
(`backend/app/core/config.py`, template in `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_PROVIDER` | `llamacpp` | `llamacpp` (local), `anthropic` (needs `RAG_ANTHROPIC_API_KEY`), `extractive` (no LLM) |
| `RAG_RETRIEVAL_MODE` | `hybrid_rerank` | `bm25`, `dense`, `hybrid`, `hybrid_rerank` |
| `RAG_TOP_K` | 5 | passages sent to the LLM |
| `RAG_RERANK_K` | 20 | fused candidates sent to the reranker |
| `RAG_MIN_RERANK_SCORE` | 0.01 | abstain before calling the LLM below this (calibrated) |
| `RAG_SKIP_RERANK_WHEN_CONFIDENT` | false | skip the reranker when BM25 and dense agree |
| `RAG_QUERY_CACHE_SIZE` | 0 | in-memory answer cache |

No secret is committed. The local model needs no key.

## Example questions

- ما هو السن القانوني للزواج في تونس؟
- ما هي إجراءات الطلاق في تونس؟
- كيف يتم تحديد حضانة الأطفال؟
- Quels sont les droits et obligations des époux ?
- Quelles sont les conditions du divorce en Tunisie ?
- What does Tunisian law say about child custody?
- ماذا كان ينص الفصل 18 في صياغته الأصلية قبل التنقيح؟ (historical wording)

## Evaluation

```bash
python scripts/build_eval_set.py                     # verify the Q&A set against the corpus
python eval/retrieval_eval.py --tag ablation         # BM25 / dense / hybrid / hybrid+rerank
python eval/rerank_scores.py && python eval/simulate_rerank.py   # threshold + rerank depth
python eval/generation_eval.py --tag baseline        # full pipeline, resumable
python eval/chunking_experiment.py
pytest -q tests
```

**Dataset** (`data/eval/qa.jsonl`): 217 questions, 139 in Arabic, 57 in
French and 21 in English. They cover factual questions, article
identification, cross-references, ambiguous questions, unanswerable
questions, hallucination bait, historical wordings and multi-step questions.
Every item has gold articles and a verbatim evidence quote; all 217 were
checked automatically against the indexed text. No lawyer has reviewed them
yet.

### Retrieval (measured, 203 answerable questions)

| Mode | Recall@1 | Recall@5 | MRR@10 | Search p95 |
|---|---|---|---|---|
| BM25 | 0.557 | 0.823 | 0.665 | 2 ms |
| Dense | 0.709 | 0.892 | 0.789 | 279 ms |
| Hybrid (RRF) | 0.660 | 0.911 | 0.765 | 274 ms |
| Hybrid + reranker | **0.842** | **0.975** | **0.900** | 7,878 ms |

### Generation and optimisation

See [`docs/evaluation.md`](docs/evaluation.md) for generation metrics,
latency and cost, and the retrieval ablation.

## Repository layout

```
backend/app/   api/ core/ ingestion/ retrieval/ generation/ models/ main.py
frontend/      React + Tailwind UI (Vite)
data/          sources.yaml, raw/ (official PDFs), ocr/ (OCR cache), eval/ (Q&A set)
eval/          evaluation scripts and results/
scripts/       fetch_models, ocr_pdf, build_index, build_eval_set, ui_smoke_test
tests/         unit and integration tests (fake models, no downloads)
design.md  docs/evaluation.md  docs/deployment.md
```

## Deployment

`docker-compose.yml` runs the backend (FastAPI, models and index in
volumes) and the frontend (nginx serving the build and proxying `/api`).
Any VM with 4 vCPUs, 16 GB of RAM and 15 GB of disk can host it. See
`docs/deployment.md`.

## Contributions

| Member | Area | Work |
|---|---|---|
| Ghada Bezine | Data & retrieval | Corpus sourcing (official IORT PDFs), OCR and article parsing, chunking, embeddings, index, BM25 / hybrid / reranking |
| Saloua Chouihi | Application | FastAPI backend, LLM integration, citation generation and checks, React frontend, UX |
| Seifallah Chourou | Evaluation & deployment | Evaluation dataset, retrieval and generation evaluation, ablations, optimisation measurements, Docker, documentation |

Implementation was carried out with an AI coding assistant (Claude Code).
