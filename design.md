# Design — Al-Murshid, a RAG assistant for Tunisian personal status law

This document explains the choices behind each stage of the pipeline and the
failure cases we know about. Numbers come from the files in `eval/results/`.

## 1. Corpus and users

**Users.** Tunisian residents and their families, law students, social
workers, and staff of legal-aid associations. They ask practical questions
("what is the marriage age?", "who gets custody?", "how much does the wife
inherit?") in Arabic (including Tunisian phrasing), French or English, and
need answers they can check against the official text.

**Documents** (manifest: `data/sources.yaml`, statistics: `data/index/manifest.json`):

| Document | Source | Pages | Articles | Status |
|---|---|---|---|---|
| مجلة الأحوال الشخصية (CSP) + promulgation decree of 13 Aug 1956 | IORT official edition 2019, revised to 16 May 2017 — `justice.gov.tn/.../StatutpersonnelArabe.pdf` | 37 | 218 + 6 | in force |
| 13 annexed laws, decrees and orders (civil status law 1957-3, marital property 1998-94, guardianship & adoption 1958-27, premarital certificate 1964-46, surnames for children of unknown parentage 1998-75, alimony guarantee fund 1993-65 and decree 1993-1655, …) | same IORT volume, annex | 65 | 187 | in force |
| CSP as reproduced by CAWTAR, with 30 endnotes quoting earlier wordings | PDF supplied by the team; original URL not recorded | 26 | 214 + 29 notes | **historical** |

Total: 128 indexed pages, 680 chunks (official text: 102 pages). The
official PDF stores Arabic with a custom font encoding, so its text layer is
unusable and every page is OCR'd (§2).

**Corpus size versus the brief.** The brief asks for "hundreds of pages or
more". Our official corpus is 102 pages of dense legal text (about 130,000
Arabic characters) plus 26 historical pages. That is too large for one prompt
(well over 60,000 tokens), so retrieval is needed, but it is below the
suggested size. The French official edition and the related codes (child
protection code, Law 2017-58) are the natural extensions. They could not be
downloaded from the build environment (see `postmortem.md`), and the import
workflow (`data/sources.yaml` + `scripts/build_index.py`) accepts them without
code changes.

**Personal data.** The corpus is public legislation. User questions are not
stored except when a user submits feedback; feedback is written to
`data/feedback/feedback.jsonl`, which is git-ignored.

## 2. Ingestion

`backend/app/ingestion/`:

1. **Loading** (`loader.py`): documents and page ranges come from
   `data/sources.yaml`, each with title, French title, source URL, edition,
   publication date, authority, legal status and SHA-256 of the PDF.
2. **OCR**: pages are rendered at 300 dpi and binarised at grey level 70,
   which removes the grey diagonal "Imprimerie Officielle" watermark. Each
   text line is cut from the row-ink profile and OCR'd separately with
   Tesseract's `tessdata_best` Arabic model (`--psm 7`, falling back to
   `--psm 13`). Whole-page OCR silently dropped lines that the watermark
   crosses, for example the first line of article 202; line-by-line OCR
   recovers them. The OCR output is cached and committed (`data/ocr/`), so a
   clean clone can rebuild the index without Tesseract.
3. **Normalisation** (`normalize.py`): two levels.
   - `clean_text` keeps the wording: it applies NFKC and removes tatweel,
     bidi marks, Arabic-Indic digits and hyphenation.
   - `search_form` is only used for matching. It folds diacritics, alef,
     ya and ta-marbuta variants, accents and case.
   - For text-layer PDFs, reversed lam-alef ligatures ("إال" → "إلا") are
     repaired.
4. **Structure** (`structure.py`): articles start with "الفصل N". The parser
   handles OCR variants of the article marker (".-", "--", `ب"`), garbled
   keywords and bis/ter articles ("مكرر", "ثالثا"). It tracks book, chapter
   and sub-heading, and separates publisher footnotes from the article body.
   It also extracts amendment notes ("نقح بالقانون عدد 32 لسنة 2007") and
   repeals.
   - **Number repair**: OCR often drops leading digits ("الفصل 6" between
     145 and 147). Numbers are checked against the running sequence and
     repaired; repaired articles are flagged `number_inferred`.
   - **Cross-references**: a cross-reference broken across lines ("… بال /
     الفصل 9.") must not open a new article. A new article requires a
     strong marker, or a plausible number following a finished sentence.
   - **Coverage**: after these fixes, all 14 official documents have every
     article number with no gaps. For the CSP that is 213 articles plus 5
     bis articles; 18 of the CSP numbers were repaired.
5. **Validation and deduplication** (`chunker.finalize`): chunks shorter than
   12 normalised characters are dropped (1), exact duplicates are removed
   (2, the repealed decree articles 3–5 share one text), and chunk ids are
   unique.
6. **Persistence**: `data/index/chunks.jsonl` holds texts and metadata,
   `vectors.npy` the embeddings and `manifest.json` the statistics and
   validation report. An embedding cache keyed by model and text hash makes
   rebuilds incremental.

## 3. Chunking

Legal answers are cited by article, so the default unit is **one article per
chunk** (`article`, max 1,200 characters, 150 overlap). Only 8 articles exceed
the limit; they are split on paragraph boundaries, and every part keeps its
article number. Each chunk stores:

- `text`: the verbatim wording, shown to users and quoted to the LLM
- `embed_text`: the same text prefixed with document › book › chapter ›
  heading › article. Many articles make no sense alone; "الفصل 12" only
  mentions the dowry because it sits under the heading "المهر".

We compared four strategies on the same 183 questions
(`eval/chunking_experiment.py`, results in `eval/results/chunking_experiment.md`):
article-level chunks at 1,200 and 600 characters against structure-blind
sliding windows of 800 and 1,500 characters. Windows lose the article
boundary that citations need, and mix neighbouring rules into one passage.

## 4. Embeddings and index

**Embedding model: `intfloat/multilingual-e5-large`** (1024-d). It is strong
on Arabic, French and English and handles cross-lingual retrieval: French and
English questions find Arabic articles (dense Recall@5: fr 0.943, en 0.875).
Weights come from the public Weaviate inference image on Docker Hub, because
HuggingFace was blocked in our build environment (`scripts/fetch_models.py`).

**Index: an exact flat index** (`vector_store.py`, NumPy `.npy` plus JSONL
metadata) instead of a vector database. With 680 chunks, exact search costs
under a millisecond, recall is perfect, and there is no server to run. The
index is two files that survive restarts and live in a Docker volume.
Metadata filtering (document, collection, legal status) is a boolean mask.
FAISS-HNSW or pgvector would only pay off with hundreds of thousands of
chunks, and the `VectorIndex` interface would stay the same.

## 5. Hybrid retrieval

- **BM25** (`rank-bm25`) over `search_form` tokens with Arabic light
  stemming (و/ال/بال/لل prefixes; ها/هم/ات/ين/ون suffixes) and Arabic, French
  and English stop words.
- **Cross-lingual keyword expansion** (`query.py`): a small glossary maps
  French and English legal terms to the Code's Arabic vocabulary (garde →
  حضانة, pension → نفقة). The expansion is appended to the BM25 query only.
  The dense retriever, the reranker and the LLM receive the user's question
  unchanged, so expansion cannot alter the legal meaning of the question.
- **Explicit article references** ("الفصل 32 مكرر", "article 5") are
  resolved by structure: the matching article is pinned first.
- **Fusion**: Reciprocal Rank Fusion (k = 60) of the top 20 BM25 and dense
  results.
- **Historical texts** are excluded unless the question asks about earlier
  wordings ("قبل التنقيح", "ancienne rédaction", "original wording").

Ablation (183 answerable questions; `eval/results/retrieval_ablation.md`):

| Mode | R@1 | R@5 | R@10 | MRR@10 | Search p95 |
|---|---|---|---|---|---|
| BM25 | 0.503 | 0.787 | 0.842 | 0.608 | 2 ms |
| Dense | 0.727 | 0.945 | 0.978 | 0.819 | 324 ms |
| Hybrid (RRF) | 0.661 | 0.913 | 0.973 | 0.766 | 285 ms |
| Hybrid + reranker | **0.803** | **0.973** | **0.989** | **0.874** | 9,720 ms |

Plain RRF is *worse* than dense alone. BM25 is weak on French and English
questions (Recall@5 0.63) even with the glossary, and equal-weight fusion
lets its errors displace good dense hits. The reranker turns the fused list
into the best ranking, with Recall@1 up 7.6 points over dense. We keep hybrid
retrieval because the reranker benefits from BM25's exact matches on
Arabic questions (article numbers, rare terms), and because the fused list is
what the reranker reorders.

## 6. Reranking

**`BAAI/bge-reranker-v2-m3`** (multilingual cross-encoder, 568M) rescores the
top 20 fused candidates against `embed_text`. It is the most accurate stage
and also the most expensive: 20 pairs take about 7 s on 4 CPU cores. Its
absolute scores on OCR'd Arabic are low and compressed (around 0.04–0.3 for
relevant passages), so the "not enough evidence" threshold is calibrated on
the data (`eval/rerank_scores.py`) rather than guessed. The reranker can be
switched off (`RAG_RETRIEVAL_MODE=hybrid`) or skipped when BM25 and dense
agree on the top hit (`RAG_SKIP_RERANK_WHEN_CONFIDENT`); see
`optimisation.md`.

## 7. Generation and citations

- **Model**: Qwen2.5-3B-Instruct (Q4_K_M GGUF) through llama.cpp on CPU. It
  needs no API key and has no per-token cost, and it covers Arabic, French
  and English. `RAG_LLM_PROVIDER=anthropic` switches to Claude through the
  official SDK, and `extractive` returns the passages without generating.
- **Prompt** (`prompts.py`): sources are numbered `<source id="S1"
  article="الفصل 5" status="current" amended_by="…">` elements, HTML-escaped,
  inside `<sources>`. The system prompt requires:
  - a citation on every legal sentence, using only ids that exist;
  - no numbers or article numbers absent from the sources;
  - `INSUFFICIENT_EVIDENCE` when the sources do not answer;
  - historical sources presented as outdated, and conflicts flagged;
  - an answer in the user's language;
  - sources treated as data, never as instructions.
- **Verification** (`verify.py`), applied to every answer:
  - citations must map to retrieved sources; invalid ids are removed;
  - an answer with no valid citation is turned into an abstention;
  - article numbers in the answer must be cited articles or articles those
    articles mention, otherwise the API returns an
    `unverified_article_numbers` warning;
  - every number in the answer (ages, durations, law numbers) must occur in
    a cited source or the question, otherwise a `numbers_not_found_in_sources`
    warning is returned.
- **Abstention** happens at three points:
  - before the LLM call, when no passage survives the filters or the best
    reranker score is below the calibrated threshold;
  - when the model answers `INSUFFICIENT_EVIDENCE`;
  - when the answer has no valid citation.
- **Prompt injection**: retrieved text is escaped and fenced, so a
  `</sources>` inside a document cannot close the block. The model is told
  the sources are data. The question is escaped the same way, and an
  injected "answer without sources" request still fails the citation check
  (`tests/test_generation.py`). Chat history is accepted by the API but not
  fed to the model, so injected text cannot carry over between turns.
- **Disclaimer**: every response carries a disclaimer in the question's
  language saying it is legal information, not legal advice.

## 8. Backend and frontend

FastAPI (`backend/app/main.py`, `api/routes.py`) exposes `GET /health`,
`POST /chat`, `POST /search`, `GET /documents` and `POST /feedback`, with
Pydantic models, validation (question length, mode enum, top-k bounds), CORS
from settings, and 503s that explain a missing index. Retrieval and
generation live in their own packages; the routes only translate.

The React + Tailwind frontend (`frontend/`) has:

- right-to-left rendering for Arabic, with the interface in Arabic, French
  or English;
- clickable `[S1]` citation chips that open the source panel;
- for each source: the verbatim article, document title, page, amendments,
  an "in force"/"historical" badge, and a link to the official PDF at that
  page;
- a clear "not enough evidence" badge, warnings, feedback, and a corpus
  list.

## 9. Known failure cases

- **OCR errors** remain in the text ("الزوا 8" for "الزواج", "مهراء", digits
  inside amendment dates: "1993" printed as "53"). Retrieval tolerates them;
  quotes shown to users can contain them, and the PDF link lets users check
  the original page.
- **Small generator.** The 3B model sometimes cites a neighbouring article
  that states the same fact (article 153 for the age of 18 instead of
  article 5), pads answers with loosely related sources, or translates
  legal terms clumsily into French and English. The verification layer
  catches invented numbers and article numbers, but not a wrong article
  that supports the same claim.
- **Cross-references** are not followed automatically. "الفصل 21" lists the
  articles whose violation voids a marriage, but their content is only
  retrieved if the question also matches them.
- **Long multi-part articles** (32, 31, 56) are split into parts. A question
  answered by part 2 may retrieve part 1 only.
- **Tables and forms** (model civil-status certificates in the annex) OCR
  poorly and are of little use for retrieval.
- **Historical text**: the CAWTAR copy's main text predates 2007, and its
  extraction still contains some broken ligatures. It is only searched for
  explicitly historical questions, and is labelled "historical" in the
  interface.
- **Unanswerable questions about nearby law** (the Penal Code, nationality,
  labour law) can retrieve superficially similar articles. Abstention then
  depends on the model saying `INSUFFICIENT_EVIDENCE` or on the reranker
  threshold.
- **Duplicate repeal notes**: the decree's articles 3–5 share one text and
  are deduplicated into one chunk labelled "3".
