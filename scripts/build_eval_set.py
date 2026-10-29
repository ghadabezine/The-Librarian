"""Verify data/eval/qa_source.yaml against the corpus and write data/eval/qa.jsonl.

Checks for every item:
  1. each gold reference "doc_id:article_label" resolves to indexed chunks
  2. the evidence quote occurs (after search normalisation) in a gold article
  3. each key-fact group has a variant present in the gold articles (Arabic or
     French version); English facts are translations and are not matched
Items failing 1 or 2 are kept but marked verified=false and reported.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.normalize import search_form  # noqa: E402
from app.ingestion.pipeline import build_chunks  # noqa: E402


def main() -> None:
    settings = get_settings()
    chunks, _ = build_chunks(settings)
    by_article: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:   # gold refs match either language version of a text (work_id)
        if c.get("article_label"):
            for key in {c["doc_id"], c.get("work_id") or c["doc_id"]}:
                by_article[f"{key}:{c['article_label']}"].append(c)

    items = yaml.safe_load((ROOT / "data/eval/qa_source.yaml").read_text(encoding="utf-8"))["items"]
    out, problems = [], []
    for it in items:
        it.setdefault("answerable", True)
        it.setdefault("facts", [])
        it.setdefault("forbid", [])
        gold_chunks = {g: by_article.get(g, []) for g in it.get("gold", [])}
        missing = [g for g, cs in gold_chunks.items() if not cs]
        text = " ".join(search_form(c["text"]) for cs in gold_chunks.values() for c in cs)
        ev_ok = bool(it.get("ev")) and search_form(it["ev"]) in text
        facts_in_source = [any(search_form(v) in text for v in group) for group in it["facts"]]
        checks = {
            "gold_resolved": not missing,
            "evidence_found": ev_ok if it["answerable"] else None,
            "facts_found_in_source": facts_in_source if it["lang"] in ("ar", "fr") else None,
        }
        if it["answerable"]:
            verified = not missing and ev_ok and (it["lang"] == "en" or all(facts_in_source))
        else:
            verified = not it.get("gold") or not missing
        if not verified:
            problems.append((it["id"], missing, ev_ok, facts_in_source))
        it["gold_chunk_ids"] = sorted(c["chunk_id"] for cs in gold_chunks.values() for c in cs)
        it["verification"] = {"auto_verified": verified, **checks, "human_legal_review": False}
        out.append(it)

    dest = ROOT / "data/eval/qa.jsonl"
    with open(dest, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_ok = sum(i["verification"]["auto_verified"] for i in out)
    print(f"{len(out)} items, {n_ok} auto-verified -> {dest}")
    print("by language:", dict(Counter(i["lang"] for i in out)))
    print("by category:", dict(Counter(i["cat"] for i in out)))
    print("answerable:", sum(i["answerable"] for i in out))
    for p in problems:
        print("UNVERIFIED", p)


if __name__ == "__main__":
    main()
