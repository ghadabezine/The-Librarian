# Postmortem

What went wrong while building the system, why, how we fixed it, and what we
would do differently. Everything here happened during this project; the
evidence is in the git history and in `eval/results/`.

## 1. The build environment could not reach the legal sources or the model hub

**What happened.** The cloud sandbox we built in only allowed package
registries, GitHub and Docker Hub. Every Tunisian legal site was refused
(legislation.tn, iort.gov.tn, justice.gov.tn, e-justice.tn), as were
refworld, Wikipedia, archive.org and HuggingFace.

**Fix.**
- **Corpus**: we could not download the official text, and we would not
  substitute an invented one. We first worked from a PDF supplied by a
  teammate (a CAWTAR reproduction). Then a teammate downloaded the official
  Ministry of Justice / IORT PDF and pushed it to our GitHub repository,
  which the sandbox could read.
- **Models**: we pulled the same public weights from Docker Hub. The
  embedder and reranker come out of Weaviate's inference images (only the
  `COPY /app/models` layer, ~1.3 GB, instead of the full 9 GB image); the
  Qwen GGUF comes from the `ai/qwen2.5` OCI artifact. `scripts/fetch_models.py`
  reproduces this with the registry API only.

**Lesson.** Check where you can download from before choosing models and
sources, and keep a reproducible fetch script, not a manual download.

## 2. The first corpus PDF was outdated

**What happened.** The first PDF looked like a consolidated text, but its
article 5 still set the marriage age at 20 for men and 17 for women. Law
2007-32 changed this to 18 for both. A RAG system built on that file would
have stated repealed law as current, with a confident citation.

**Fix.** The official IORT 2019 edition (revised to 16 May 2017, including
the 2007 and 2010 amendments) became the main source. The CAWTAR copy is
kept only as a `historical` document: its endnotes quote the earlier
wordings. It is searched only when the question asks about history, and it
is labelled "historical — no longer in force" in the interface.

**Lesson.** Check the version of a legal source against a known amendment
before indexing it. "Consolidated" on a PDF tells you nothing about the date.

## 3. The official PDF had no usable text layer

**What happened.** PyMuPDF extracted the IORT PDF as mojibake: the Arabic
glyphs use a custom font encoding without a Unicode map.

**Fix.** OCR with Tesseract and the `tessdata_best` Arabic model, from
GitHub. A diagonal grey watermark covers every page; a grey-level threshold
removes most of it.

## 4. OCR silently dropped whole lines

**What happened.** The first article-coverage check showed gaps: CSP
articles 14, 17, 93, 181 and 202 were missing, along with several articles
of the civil-status law. For article 202 the whole first line was gone, not
just misread. Where the watermark crosses a line, Tesseract's page layout
analysis discarded the line. Even a clean crop of that line returned
nothing in line mode (`--psm 7`) and only worked in raw-line mode
(`--psm 13`).

**Fix.** We segment lines ourselves from the row-ink profile and OCR each
band (`--psm 7`, falling back to `--psm 13`). We also lowered the threshold
from 110 to 70, because the watermark's darkest pixels sit at 80–110.

**Lesson.** A completeness check (every article number from 1 to N is
present) turned an invisible data-loss bug into a visible one. Validate
extraction against the document's own structure.

## 5. OCR misread article numbers, and cross-references looked like articles

**What happened.** Bold article numbers lost leading digits ("الفصل 6" for
146, "الفصل 0" for 210). A line break inside a cross-reference ("…المنصوص
عليها بال" / "الفصل 9.") opened a phantom article 9 inside article 13.

**Fix.** We check numbers against the running sequence and repair dropped
leading digits (flagged `number_inferred`, 18 in the CSP). A new article now
needs either a strong marker (".-", "(", or `ب"`, which is how OCR renders
".-") or a plausible number after a finished sentence. These cases are unit
tests now (`tests/test_structure.py`).

## 6. A footnote word triggered the "repealed" flag

**What happened.** Article 5 was marked as repealed. An IORT editor's
footnote at the page break ("(*) في النسخة الأصلية…") had been merged into
the article, and "النسخة" contains "نسخ" (repealed).

**Fix.** Footnotes are kept apart, marked `[هامش]`, and repeal keywords must
match as whole words.

## 7. A test overwrote the real corpus manifest

**What happened.** The API test for `/documents` wrote `documents: []` into
the real `data/sources.yaml`. `sources_file` defaulted to the repository
path instead of the test's temporary `data_dir`, and the file was not
committed yet. A second test wrote two feedback records into the real
`data/feedback/`.

**Fix.** We restored the file and made `sources_file` a property derived
from `data_dir`, so tests cannot reach repository data. We fixed the
dependency override that keyed on a monkeypatched function, and committed
earlier.

**Lesson.** Every path in settings should derive from one root that tests
override, and commit data files as soon as they exist.

## 8. Hybrid retrieval was worse than dense retrieval

**What happened.** We expected BM25 + dense with RRF to beat dense alone.
It did not: Recall@5 was 0.913 against 0.945. BM25 cannot match French or
English questions to Arabic text (Recall@5 0.63) even with a glossary, and
equal-weight fusion let its misses push good dense hits down.

**Response.** We report the result as measured and did not tune the
benchmark until hybrid won. The reranker recovers the loss (0.973), and the
fused list is still the better candidate pool for it. Possible next steps:
language-aware fusion weights, or skipping BM25 for non-Arabic questions.

## 9. The reranker's scores were miscalibrated for our default threshold

**What happened.** We had set the "not enough evidence" threshold to 0.05
by intuition. A spot check showed the correct article scoring 0.039, so
that threshold could refuse answerable questions.

**Fix.** We recorded every (question, candidate) reranker score once
(`eval/rerank_scores.py`) and chose the threshold from the data. At 0.01,
2.7% of answerable questions are refused and 64% of unanswerable questions
are caught before the LLM call. The same recording let us study rerank
depth offline. Caveat: the threshold is tuned on the evaluation set itself;
a separate dev set would be cleaner.

## 10. CPU-only generation made evaluation slow, and the container restarted

**What happened.**
- Each full answer took about 45–55 s on 4 CPU cores (Qwen2.5-3B Q4,
  ~1,300–1,500 prompt tokens); the reranker alone took ~7 s.
- The cloud container restarted several times during long jobs, killing
  them.
- One embedding run took 6 minutes per batch because llama.cpp was
  compiling in parallel.

**Fix.**
- Evaluation scripts are resumable: rows already written are skipped.
- Long jobs run one at a time so timings are not distorted.
- The embedding cache makes index rebuilds incremental (50 s instead of
  ~10 min after a parser fix).

## 11. Smaller problems

- Container image and model downloads were slow (~1 MB/s); torch with CUDA
  wheels is 3 GB. The Dockerfile installs the CPU-only torch wheel.
- `pkill -f <pattern>` killed our own shell when the pattern appeared in
  the same command line.
- The CAWTAR text layer reverses lam-alef ligatures ("إال" for "إلا");
  only unambiguous patterns are repaired.

## What we would do differently

1. Verify source availability, provenance and version on day one, before
   any code.
2. Build the article-coverage check first. It found every serious
   extraction bug.
3. Hold out a development split of the questions for calibrating
   thresholds and choosing optimisation settings.
4. Budget for CPU inference time in the evaluation plan, or run the
   generator on a GPU or API for the evaluation runs.
