"""Build (or rebuild) the search index from data/sources.yaml.

    python scripts/build_index.py              # chunks + embeddings
    python scripts/build_index.py --no-embed   # chunks only (BM25 works, dense does not)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.pipeline import build_chunks, write_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--strategy", choices=["article", "fixed"])
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    settings = get_settings()
    chunks, report = build_chunks(settings, args.strategy)
    for d in report["documents"]:
        print(f"{d['doc_id']:32s} pages={d['pages']:3d} articles={d['articles']:3d} chunks={d['chunks']:3d} "
              f"missing={d['missing_article_numbers']} inferred={len(d['inferred_article_numbers'])}")
    print(f"total: {report['total_pages']} pages, {report['total_chunks']} chunks; "
          f"dropped {len(report['validation']['dropped_short'])} short, "
          f"{len(report['validation']['duplicates'])} duplicates")
    write_index(settings, chunks, report, embed=not args.no_embed)
    print("index written to", settings.index_dir)


if __name__ == "__main__":
    main()
