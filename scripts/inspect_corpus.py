import json
import sys
from pathlib import Path
from collections import Counter

QA_PATH = Path("data/eval/qa.jsonl")
QA_SRC_PATH = Path("data/eval/qa_source.yaml")
SOURCES_PATH = Path("data/sources.yaml")
OCR_DIR = Path("data/ocr/csp_ar_iort_2019")


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# ---------- 1. Existing qa.jsonl ----------
section(f"1. EXISTING QA SET: {QA_PATH}")
if not QA_PATH.exists():
    print("NOT FOUND")
else:
    lines = [l for l in QA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"bytes: {QA_PATH.stat().st_size:,}")
    print(f"non-empty lines: {len(lines)}")

    schemas = Counter()
    bad = 0
    for l in lines:
        try:
            r = json.loads(l)
            schemas[tuple(sorted(r.keys()))] += 1
        except json.JSONDecodeError:
            bad += 1
    print(f"unparseable lines: {bad}")

    if lines:
        try:
            first = json.loads(lines[0])
            print("\nFirst row (pretty):")
            print(json.dumps(first, indent=2, ensure_ascii=False)[:1500])
        except json.JSONDecodeError:
            print(f"\nFirst line is NOT valid JSON. First 300 chars:\n{lines[0][:300]}")

    if schemas:
        print("\nField schemas seen:")
        for ks, c in schemas.most_common(5):
            print(f"  {c:>4} rows -> {list(ks)}")


# ---------- 2. QA source yaml ----------
section(f"2. QA SOURCE: {QA_SRC_PATH}")
if not QA_SRC_PATH.exists():
    print("NOT FOUND")
else:
    txt = QA_SRC_PATH.read_text(encoding="utf-8")
    print(f"bytes: {QA_SRC_PATH.stat().st_size:,}")
    print(f"first 800 chars:\n{txt[:800]}")


# ---------- 3. Corpus manifest ----------
section(f"3. SOURCES MANIFEST: {SOURCES_PATH}")
if not SOURCES_PATH.exists():
    print("NOT FOUND")
else:
    txt = SOURCES_PATH.read_text(encoding="utf-8")
    print(f"bytes: {SOURCES_PATH.stat().st_size:,}")
    print(f"first 500 chars:\n{txt[:500]}")


# ---------- 4. OCR pages ----------
section(f"4. OCR PAGES: {OCR_DIR}")
if not OCR_DIR.exists():
    print("NOT FOUND")
else:
    pages = sorted(OCR_DIR.glob("page_*.txt"))
    print(f"total pages: {len(pages)}")
    sizes = [p.stat().st_size for p in pages]
    if sizes:
        print(f"min/median/max bytes: {min(sizes)} / {sorted(sizes)[len(sizes)//2]} / {max(sizes)}")
    empties = [p.name for p in pages if p.stat().st_size < 50]
    print(f"near-empty (<50 bytes): {len(empties)} -> {empties}")

    for name in ["page_003.txt", "page_010.txt", "page_030.txt"]:
        p = OCR_DIR / name
        if p.exists():
            print(f"\n--- {name} (first 700 chars) ---")
            print(p.read_text(encoding="utf-8")[:700])