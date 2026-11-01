import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

QA = Path("data/eval/qa.jsonl")
SHEET = Path("data/eval/review_sheet.csv")
BACKUP = QA.with_suffix(f".jsonl.bak-{datetime.now():%Y%m%d-%H%M%S}")

VALID_DECISIONS = {"approve", "reject", "fix"}

if not SHEET.exists():
    raise SystemExit(f"{SHEET} not found. Run make_review_sheet.py first.")

with SHEET.open(encoding="utf-8-sig", newline="") as f:
    review = list(csv.DictReader(f))

errors = []
for i, r in enumerate(review, start=2):
    d = (r.get("decision") or "").strip().lower()
    if d not in VALID_DECISIONS:
        errors.append(f"row {i} ({r.get('id')}): decision '{d}' not in {VALID_DECISIONS}")
    if d == "approve" and not (r.get("reviewer") or "").strip():
        errors.append(f"row {i} ({r.get('id')}): approve needs a reviewer name")
    if d == "approve" and not (r.get("reviewed_at") or "").strip():
        errors.append(f"row {i} ({r.get('id')}): approve needs reviewed_at (YYYY-MM-DD)")
    if d in {"reject", "fix"} and not (r.get("notes") or "").strip():
        errors.append(f"row {i} ({r.get('id')}): {d} needs notes")

if errors:
    print("\n".join(errors[:40]))
    print(f"\n{len(errors)} problem(s) — nothing written. Fix and re-run.")
    raise SystemExit(1)

decisions = {r["id"]: r for r in review}

rows = [json.loads(l) for l in QA.read_text(encoding="utf-8").splitlines() if l.strip()]
touched = {"approve": 0, "reject": 0, "fix": 0}
skipped = []

for r in rows:
    rid = r["id"]
    if rid not in decisions:
        skipped.append(rid)
        continue
    d = decisions[rid]["decision"].strip().lower()
    v = r.setdefault("verification", {})
    v["human_legal_review"] = (d == "approve")
    v["reviewed_by"] = decisions[rid]["reviewer"].strip()
    v["reviewed_at"] = decisions[rid]["reviewed_at"].strip()
    v["review_decision"] = d
    note = (decisions[rid].get("notes") or "").strip()
    if note:
        v["review_notes"] = note
    touched[d] += 1

shutil.copyfile(QA, BACKUP)
with QA.open("w", encoding="utf-8", newline="") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Backup:   {BACKUP}")
print(f"Updated:  {QA}")
print(f"Approve:  {touched['approve']}")
print(f"Reject:   {touched['reject']}  (kept, marked human_legal_review=false)")
print(f"Fix:      {touched['fix']}  (kept; fix qa_source.yaml and rebuild)")
print(f"Untouched rows not in review sheet: {len(skipped)}")