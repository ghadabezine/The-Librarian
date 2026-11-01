import json
import sys
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval/qa_pairs.jsonl")
REQUIRED = ["id", "question", "reference_answer", "category",
            "difficulty", "answerable", "split", "source_passages"]
VALID_SPLIT = {"dev", "test"}
VALID_DIFF = {"single_hop", "multi_hop"}


def is_placeholder(v) -> bool:
    s = str(v).strip()
    return (
        not s
        or set(s) <= {"."}
        or s.upper().startswith("[REPLACE")
    )


errors: list[str] = []
ids: dict[str, int] = {}
count = 0

for n, raw in enumerate(PATH.read_text(encoding="utf-8").splitlines(), start=1):
    if not raw.strip():
        continue
    count += 1

    try:
        r = json.loads(raw)
    except json.JSONDecodeError as e:
        errors.append(f"{PATH}:{n}:{e.colno}  {e.msg}")
        errors.append("    " + raw.strip()[:140])
        errors.append("    " + " " * min(e.colno - 1, 139) + "^")
        continue

    if not isinstance(r, dict):
        errors.append(f"{PATH}:{n}  row is not a JSON object")
        continue

    for k in REQUIRED:
        if k not in r:
            errors.append(f"{PATH}:{n}  missing field '{k}'")

    for k in ("id", "question", "reference_answer", "category"):
        if is_placeholder(r.get(k, "")):
            errors.append(f"{PATH}:{n}  '{k}' is empty or a placeholder")

    rid = r.get("id")
    if rid is not None:
        if rid in ids:
            errors.append(f"{PATH}:{n}  duplicate id '{rid}' (first seen line {ids[rid]})")
        else:
            ids[rid] = n

    if r.get("split") not in VALID_SPLIT:
        errors.append(f"{PATH}:{n}  bad split {r.get('split')!r}")
    if r.get("difficulty") not in VALID_DIFF:
        errors.append(f"{PATH}:{n}  bad difficulty {r.get('difficulty')!r}")
    if not isinstance(r.get("answerable"), bool):
        errors.append(f"{PATH}:{n}  'answerable' must be true/false")

    sp = r.get("source_passages")
    if not isinstance(sp, list):
        errors.append(f"{PATH}:{n}  'source_passages' must be a list")
    elif r.get("answerable") and not sp:
        errors.append(f"{PATH}:{n}  answerable row has no source_passages")
    else:
        for i, p in enumerate(sp):
            if not isinstance(p, dict):
                errors.append(f"{PATH}:{n}  source_passages[{i}] is not an object")
                continue
            for k in ("article", "text"):
                if is_placeholder(p.get(k, "")):
                    errors.append(f"{PATH}:{n}  source_passages[{i}].{k} is empty or placeholder")

if errors:
    print("\n".join(errors))
    print(f"\n{len(errors)} problem(s) in {count} rows")
    sys.exit(1)

rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
by_split: dict[str, int] = {}
for r in rows:
    by_split[r["split"]] = by_split.get(r["split"], 0) + 1

print(f"{count} rows OK")
print(f"splits: {by_split}")
print(f"answerable: {sum(1 for r in rows if r['answerable'])} / {len(rows)}")