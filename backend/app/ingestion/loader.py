"""Load source documents listed in data/sources.yaml into per-page text.

Two extraction paths:

* ``text`` – the PDF has a usable text layer (PyMuPDF ``get_text``).
* ``ocr``  – the text layer is unusable (the official IORT edition embeds Arabic
  with a custom font encoding, which extracts as garbage). Pages are rendered
  at 300 dpi, the grey diagonal watermark is removed by thresholding and each
  text line is OCR'd with Tesseract (``tessdata_best`` Arabic model). OCR output is
  cached per page under ``data/ocr/<doc_id>/`` and committed, so indexing is
  reproducible without Tesseract installed.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .normalize import clean_text, fix_lam_alef

log = logging.getLogger(__name__)


@dataclass
class SourceDoc:
    doc_id: str
    file: str
    title: str
    language: str
    extraction: str = "text"
    source_url: str | None = None
    publisher: str | None = None
    edition: str | None = None
    publication_date: str | None = None
    authority: str = "official"           # official | secondary
    legal_status: str = "consolidated"    # consolidated | historical | complementary
    version_note: str | None = None
    sha256: str | None = None
    pages: list[int] | None = None        # 1-based [first, last] PDF pages to index
    notes_pages: list[int] | None = None  # pages holding endnotes (historical wording)
    ocr_cache: str | None = None          # data/ocr/<ocr_cache>/ (shared by sub-documents of one PDF)
    title_fr: str | None = None
    collection: str = "core"              # core | annex | historical
    extra: dict = field(default_factory=dict)


@dataclass
class Page:
    doc_id: str
    page: int        # 1-based PDF page number
    text: str


def load_sources(sources_file: Path) -> list[SourceDoc]:
    data = yaml.safe_load(sources_file.read_text(encoding="utf-8"))
    known = SourceDoc.__dataclass_fields__.keys()
    docs = []
    for entry in data["documents"]:
        entry = dict(entry)
        extra = {k: v for k, v in entry.items() if k not in known}
        docs.append(SourceDoc(**{k: v for k, v in entry.items() if k in known}, extra=extra))
    return docs


def _bands(img, min_height: int = 18, min_ink: int = 2) -> list[tuple[int, int]]:
    """Horizontal text-line bands from the row ink profile of a binary page."""
    ink = (img == 0).sum(1)
    bands, start = [], None
    for i, v in enumerate(ink):
        if v > min_ink and start is None:
            start = i
        elif v <= min_ink and start is not None:
            if i - start >= min_height:
                bands.append((start, i))
            start = None
    return bands


def ocr_page(pdf_path: Path, page_index: int, tessdata_dir: Path | None, langs: str = "ara",
             threshold: int = 70) -> str:
    """OCR one page line by line.

    Whole-page Tesseract (psm 6) silently dropped some lines that the
    watermark crosses (e.g. the first line of CSP article 202). We therefore
    cut the binarised page into text-line bands and OCR each band with psm 7,
    falling back to psm 13 (raw line) when psm 7 returns nothing.
    """
    import numpy as np
    import pymupdf

    doc = pymupdf.open(pdf_path)
    pix = doc[page_index].get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    # Body text is near-black; the diagonal watermark is grey (> ~80).
    img = np.where(img < threshold, 0, 255).astype(np.uint8)
    lines = []
    for top, bottom in _bands(img):
        crop = np.ascontiguousarray(img[max(0, top - 8): bottom + 8])
        png = pymupdf.Pixmap(pymupdf.csGRAY, crop.shape[1], crop.shape[0], crop.tobytes(), False).tobytes("png")
        text = ""
        # Tall bands are figures or several touching lines: use block mode.
        for psm in (("7", "13") if bottom - top <= 110 else ("6",)):
            cmd = ["tesseract", "stdin", "stdout", "-l", langs, "--psm", psm]
            if tessdata_dir:
                cmd += ["--tessdata-dir", str(tessdata_dir)]
            text = subprocess.run(cmd, input=png, capture_output=True, check=True, timeout=120,
                                  env={**os.environ, "OMP_THREAD_LIMIT": "1"}).stdout.decode("utf-8").strip()
            if text:
                break
        if text:
            lines.append(text)
    return "\n".join(lines) + "\n"


def load_pages(src: SourceDoc, data_dir: Path, tessdata_dir: Path | None = None) -> list[Page]:
    import pymupdf

    pdf_path = data_dir / "raw" / src.file
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} missing; see data/README.md for how to obtain it")
    n_pages = pymupdf.open(pdf_path).page_count
    first, last = src.pages or [1, n_pages]
    pages = []
    for p in range(first, last + 1):
        if src.extraction == "ocr":
            cache = data_dir / "ocr" / (src.ocr_cache or src.doc_id) / f"page_{p:03d}.txt"
            if cache.exists():
                raw = cache.read_text(encoding="utf-8")
            else:
                log.info("OCR %s page %d", src.doc_id, p)
                raw = ocr_page(pdf_path, p - 1, tessdata_dir)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(raw, encoding="utf-8")
        else:
            raw = fix_lam_alef(pymupdf.open(pdf_path)[p - 1].get_text())
        pages.append(Page(src.doc_id, p, clean_text(raw)))
    return pages
