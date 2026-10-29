"""End-to-end ingestion: sources.yaml -> pages -> units -> chunks -> index.

Outputs (data/index/):
  chunks.jsonl       one JSON object per chunk (text + metadata)
  vectors.npy        dense embeddings, row i <-> line i of chunks.jsonl
  vectors_meta.json  embedding model name and chunk-id order
  manifest.json      corpus statistics and validation report
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..core.config import Settings
from .chunker import chunk_article_units, chunk_fixed, finalize
from .loader import load_pages, load_sources
from .structure import parse_units, parse_units_fr

log = logging.getLogger(__name__)


def build_chunks(settings: Settings, strategy: str | None = None, max_chars: int | None = None,
                 overlap: int | None = None) -> tuple[list[dict], dict]:
    strategy = strategy or settings.chunk_strategy
    max_chars = max_chars or settings.chunk_max_chars
    overlap = settings.chunk_overlap_chars if overlap is None else overlap
    all_chunks, report = [], {"documents": [], "strategy": strategy, "max_chars": max_chars, "overlap": overlap}
    for src in load_sources(settings.sources_file):
        pages = load_pages(src, settings.data_dir)
        if src.language == "fr":
            units = parse_units_fr(src.doc_id, pages, allow_restart=bool(src.extra.get("multi_act")))
        else:
            units = parse_units(src.doc_id, pages, src.notes_pages, trust_numbers=src.extraction == "text",
                                allow_restart=bool(src.extra.get("multi_act")))
        if strategy == "fixed":
            chunks = chunk_fixed(src, units, max_chars, overlap)
        else:
            chunks = chunk_article_units(src, units, max_chars, overlap)
        arts = [u for u in units if u.unit_type == "article"]
        nums = sorted({u.article_number for u in arts if u.article_number})
        report["documents"].append({
            "doc_id": src.doc_id, "work_id": src.work_id or src.doc_id, "language": src.language,
            "title": src.title, "legal_status": src.legal_status,
            "authority": src.authority, "pages": len(pages),
            "page_range": [pages[0].page, pages[-1].page] if pages else None,
            "characters": sum(len(p.text) for p in pages),
            "articles": len(arts), "notes": sum(u.unit_type == "note" for u in units),
            "repealed_articles": sum(u.repealed for u in arts),
            "inferred_article_numbers": [u.article_label for u in arts if u.number_inferred],
            "missing_article_numbers": ([n for n in range(1, nums[-1] + 1) if n not in nums]
                                        if nums and not src.extra.get("multi_act") else []),
            "chunks": len(chunks),
        })
        all_chunks.extend(chunks)
    chunks, validation = finalize(all_chunks)
    report["validation"] = validation
    report["total_chunks"] = len(chunks)
    report["total_pages"] = sum(d["pages"] for d in report["documents"])
    return chunks, report


def _embed_with_cache(embedder, texts: list[str], index_dir: Path):
    """Embed texts, reusing vectors of unchanged texts from earlier builds.

    The cache (embed_cache.npz) is keyed by sha1(model name + text), so a
    rebuild after a parser fix only embeds the chunks whose text changed.
    """
    import hashlib

    import numpy as np

    path = index_dir / "embed_cache.npz"
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        data = np.load(path)
        cache = dict(zip(data["keys"].tolist(), data["vectors"]))
    keys = [hashlib.sha1((embedder.name + "\x00" + t).encode()).hexdigest() for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        fresh = embedder.embed_passages([texts[i] for i in todo])
        for i, v in zip(todo, fresh):
            cache[keys[i]] = v
    vecs = np.stack([cache[k] for k in keys]).astype(np.float32)
    live = set(keys)
    kept = [k for k in cache if k in live]
    np.savez(path, keys=np.array(kept), vectors=np.stack([cache[k] for k in kept]))
    return vecs, len(texts) - len(todo)


def write_index(settings: Settings, chunks: list[dict], report: dict, embed: bool = True) -> None:
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    if embed:
        from ..retrieval.embedder import get_embedder
        from ..retrieval.vector_store import VectorIndex

        embedder = get_embedder(str(settings.models_dir / settings.embedding_model), settings.device)
        t = time.perf_counter()
        vecs, reused = _embed_with_cache(embedder, [c["embed_text"] for c in chunks], index_dir)
        report["embedding_seconds"] = round(time.perf_counter() - t, 1)
        report["embeddings_reused_from_cache"] = reused
        VectorIndex(vecs, [c["chunk_id"] for c in chunks], embedder.name).save(index_dir)
    report["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["embedding_model"] = settings.embedding_model
    (index_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
