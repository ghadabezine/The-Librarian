"""Turn structural units into retrieval chunks.

Strategies (chosen with RAG_CHUNK_STRATEGY, compared in eval/chunking_experiment.py):

* ``article`` (default) – one chunk per article. Articles longer than
  ``max_chars`` are split on paragraph boundaries with ``overlap_chars`` of
  overlap; every piece keeps the article number, so a citation always points
  to a real article.
* ``fixed`` – baseline: the document text is cut into sliding windows of
  ``max_chars`` characters regardless of structure. Each window is labelled
  with the articles it overlaps.

Each chunk carries two texts: ``text`` (verbatim, shown to users and quoted
by the LLM) and ``embed_text`` (text prefixed with document / book / chapter
/ article context, used for embedding and reranking, because many articles
make no sense without their heading, e.g. "المهر" + "الفصل 12").
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict

from .loader import SourceDoc
from .normalize import search_form
from .structure import Unit


def _context(src: SourceDoc, u: Unit) -> str:
    parts = [src.title]
    for p in (u.book, u.section, u.heading):
        if p:
            parts.append(p)
    if u.unit_type == "article":
        parts.append(f"الفصل {u.article_label}")
    elif u.unit_type == "note":
        parts.append(f"هامش [{u.note_number}] (صيغة سابقة / ملاحظة تاريخية)")
    return " › ".join(parts)


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paras = [p for p in re.split(r"\n+", text) if p.strip()]
    pieces, cur = [], ""
    for p in paras:
        while len(p) > max_chars:      # a single huge paragraph: cut on sentence ends
            cut = max(p.rfind(".", 0, max_chars), p.rfind("؛", 0, max_chars), p.rfind("،", 0, max_chars))
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(p[:cut])
            p = p[cut:].strip()
        if cur and len(cur) + len(p) + 1 > max_chars:
            pieces.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = (tail + "\n" + p) if tail else p
        else:
            cur = (cur + "\n" + p) if cur else p
    if cur:
        pieces.append(cur)
    return pieces


def _base(src: SourceDoc) -> dict:
    return {
        "doc_id": src.doc_id, "doc_title": src.title, "doc_title_fr": src.title_fr,
        "source_url": src.source_url, "publisher": src.publisher, "edition": src.edition,
        "publication_date": src.publication_date, "language": src.language,
        "authority": src.authority, "legal_status": src.legal_status, "collection": src.collection,
        "version_note": src.version_note,
    }


def _slug(label: str | None) -> str:
    return (label or "x").replace(" مكرر", "bis").replace(" ", "_")


def chunk_article_units(src: SourceDoc, units: list[Unit], max_chars: int, overlap: int) -> list[dict]:
    chunks = []
    counters: dict[str, int] = {}
    for u in units:
        pieces = _split_long(u.text, max_chars, overlap)
        ctx = _context(src, u)
        if u.unit_type == "article":
            key = f"art{_slug(u.article_label)}"
        elif u.unit_type == "note":
            key = f"note{u.note_number}"
        else:
            key = "pre"
        counters[key] = counters.get(key, 0) + 1
        if counters[key] > 1:          # same article twice (e.g. OCR repair collision)
            key = f"{key}-{counters[key]}"
        for n, piece in enumerate(pieces, 1):
            meta = asdict(u)
            meta.pop("text")
            chunk = {
                **_base(src), **meta,
                "chunk_id": f"{src.doc_id}:{key}:{n}",
                "chunk_type": u.unit_type,
                "article_label": u.article_label,
                "part": n, "parts": len(pieces),
                "text": piece,
                "embed_text": ctx + (f" (جزء {n}/{len(pieces)})" if len(pieces) > 1 else "") + "\n" + piece,
                "context": ctx,
            }
            chunks.append(chunk)
    return chunks


def chunk_fixed(src: SourceDoc, units: list[Unit], max_chars: int, overlap: int) -> list[dict]:
    """Structure-blind sliding windows (baseline for the chunking experiment)."""
    text, spans = "", []
    for u in units:
        start = len(text)
        text += u.text + "\n"
        spans.append((start, len(text), u))
    chunks, pos, n = [], 0, 0
    step = max(1, max_chars - overlap)
    while pos < len(text):
        window = text[pos: pos + max_chars]
        covered = [u for s, e, u in spans if s < pos + max_chars and e > pos]
        arts = [u for u in covered if u.unit_type == "article"]
        first = arts[0] if arts else (covered[0] if covered else None)
        n += 1
        chunks.append({
            **_base(src),
            "chunk_id": f"{src.doc_id}:win{n}",
            "chunk_type": "article" if arts else "preamble",
            "article_number": first.article_number if first else None,
            "article_suffix": first.article_suffix if first else None,
            "article_label": first.article_label if first else None,
            "covered_articles": [u.article_label for u in arts],
            "page_start": covered[0].page_start if covered else None,
            "page_end": covered[-1].page_end if covered else None,
            "book": first.book if first else None, "section": first.section if first else None,
            "heading": first.heading if first else None, "amendments": [], "amending_laws": [],
            "repealed": False, "number_inferred": False, "note_number": None, "mentions": [],
            "part": 1, "parts": 1, "text": window.strip(),
            "embed_text": src.title + "\n" + window.strip(), "context": src.title,
        })
        pos += step
    return chunks


def finalize(chunks: list[dict]) -> tuple[list[dict], dict]:
    """Validate, deduplicate and add the search form. Returns (chunks, report)."""
    seen: dict[str, str] = {}
    out, dropped_short, duplicates = [], [], []
    for c in chunks:
        c["search_text"] = c["embed_text"]
        norm = search_form(c["text"])
        if len(norm) < 12:
            dropped_short.append(c["chunk_id"])
            continue
        h = hashlib.sha1(norm.encode()).hexdigest()
        if h in seen:
            duplicates.append((c["chunk_id"], seen[h]))
            continue
        seen[h] = c["chunk_id"]
        c["text_hash"] = h
        out.append(c)
    ids = [c["chunk_id"] for c in out]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    return out, {"dropped_short": dropped_short, "duplicates": duplicates}
