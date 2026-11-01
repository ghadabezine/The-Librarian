import csv
import json
from pathlib import Path

QA = Path("data/eval/qa.jsonl")
SELECTION = Path("data/eval/review_selection.txt")
OUT = Path("data/eval/review_sheet.csv")

rows = {r["id"]: r for r in (json.loads(l) for l in QA.read_text(encoding="utf-8").splitlines() if l.strip())}
selected_ids = [l.strip() for l in SELECTION.read_text(encoding="utf-8").splitlines() if l.strip()]

missing = [i for i in selected_ids if i not in rows]
if missing:
    raise SystemExit(f"Selection references ids not in qa.jsonl: {missing}")

with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL)
    w.writerow([
        "id", "lang", "cat", "diff",
        "question", "answer", "evidence", "gold",
        "decision", "reviewer", "reviewed_at", "notes",
    ])
    for rid in selected_ids:
        r = rows[rid]
        w.writerow([
            rid,
            r.get("lang", ""),
            r.get("cat", ""),
            r.get("diff", ""),
            r.get("q", ""),
            r.get("ans", ""),
            r.get("ev", ""),
            "; ".join(r.get("gold", [])),
            "", "", "", "",
        ])

print(f"Wrote {OUT} with {len(selected_ids)} rows")
print("Reviewer instructions:")
print("  - Fill 'decision' with: approve | reject | fix")
print("  - Fill 'reviewer' with your name")
print("  - Fill 'reviewed_at' with YYYY-MM-DD")
print("  - Fill 'notes' with the reason if decision != approve")
print("  - Save as: CSV UTF-8 (do NOT save as .xlsx)")