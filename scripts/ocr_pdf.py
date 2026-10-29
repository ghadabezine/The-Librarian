"""OCR the PDF pages of documents with ``extraction: ocr`` into data/ocr/.

The OCR cache is committed to the repository, so you only need this script
(and Tesseract + the tessdata_best Arabic model) to regenerate it:

    apt-get install tesseract-ocr
    curl -L -o tessdata/ara.traineddata \\
      https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/ara.traineddata
    python scripts/ocr_pdf.py --tessdata tessdata [--force] [--workers 3]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ingestion.loader import load_sources, ocr_page  # noqa: E402


def _job(args):
    pdf, page, out, tessdata = args
    out.write_text(ocr_page(pdf, page - 1, tessdata), encoding="utf-8")
    return page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tessdata", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    import pymupdf

    jobs = {}
    for src in load_sources(ROOT / "data" / "sources.yaml"):
        if src.extraction != "ocr":
            continue
        pdf = ROOT / "data" / "raw" / src.file
        cache = ROOT / "data" / "ocr" / (src.ocr_cache or src.doc_id)
        cache.mkdir(parents=True, exist_ok=True)
        for page in range(1, pymupdf.open(pdf).page_count + 1):
            out = cache / f"page_{page:03d}.txt"
            if args.force or not out.exists():
                jobs[(pdf, page)] = (pdf, page, out, args.tessdata)
    with ProcessPoolExecutor(args.workers) as ex:
        for page in ex.map(_job, jobs.values()):
            print("ocr page", page, flush=True)


if __name__ == "__main__":
    main()
