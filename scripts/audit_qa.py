import json
from pathlib import Path
from collections import Counter, defaultdict

QA = Path("data/eval/qa.jsonl")
OCR = Path("data/ocr/csp_ar_iort_2019")
OUT_CSV = Path("data/eval/qa_audit.csv")


def hdr(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


rows = [json.loads(l) for l in QA.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Loaded {len(rows)} rows from {QA}")

# ---- 1. Distribution ----
hdr("1. DISTRIBUTION")
for field in ("lang", "cat", "diff"):
    print(f"  {field:<6} {dict(Counter(r.get(field) for r in rows))}")
print(f"  answerable: {sum(1 for r in rows if r.get('answerable'))} / {len(rows)}")

# ---- 2. Verification flags ----
hdr("2. VERIFICATION FLAG COUNTS")
for f in ("auto_verified", "gold_resolved", "evidence_found", "human_legal_review"):
    print(f"  {f:<22} {dict(Counter(r.get('verification', {}).get(f) for r in rows))}")

# ---- 3. Split pools: positive-verifiable vs refusal test ----
REFUSAL_CATS = {"unanswerable", "hallucination_bait"}


def facts_ok(r):
    facts = r.get("verification", {}).get("facts_found_in_source")
    if not facts:
        return True  # nothing to check -> not a failure
    return all(x is not False for x in facts)


def is_refusal_case(r):
    return (
        not r.get("answerable")
        or r.get("cat") in REFUSAL_CATS
        or not str(r.get("ev", "")).strip()
    )


def passes_auto(r):
    v = r.get("verification", {})
    if is_refusal_case(r):
        return True  # refusal cases don't need evidence to "pass"
    return (
        v.get("auto_verified") is True
        and v.get("gold_resolved") is True
        and v.get("evidence_found") is True
        and facts_ok(r)
    )


pass_auto = [r for r in rows if passes_auto(r)]
fail_auto = [r for r in rows if not passes_auto(r)]
positives = [r for r in pass_auto if not is_refusal_case(r)]

hdr("3. AUTO-VERIFICATION PASS (ignoring human_legal_review)")
print(f"  PASS total:            {len(pass_auto)}")
print(f"  of which refusal test: {len(pass_auto) - len(positives)}")
print(f"  POSITIVE candidates:   {len(positives)}   <- pool for the 100 verified")
print(f"  FAIL:                  {len(fail_auto)}")

# ---- 4. Failure reasons ----
hdr("4. FAILURE REASONS (positive pool only)")
reasons = Counter()
for r in fail_auto:
    if is_refusal_case(r):
        continue
    v = r.get("verification", {})
    if not v.get("gold_resolved"):
        reasons["gold_resolved=false"] += 1
    if not v.get("evidence_found"):
        reasons["evidence_found=false"] += 1
    if not facts_ok(r):
        reasons["facts_found_in_source has false"] += 1
for reason, n in reasons.most_common():
    print(f"  {n:>4}  {reason}")
if not reasons:
    print("  (none)")

# ---- 5. Refusal pool (for faithfulness testing) ----
hdr("5. REFUSAL / UNANSWERABLE POOL")
refusals = [r for r in rows if is_refusal_case(r)]
print(f"  count: {len(refusals)}")
for r in refusals:
    print(f"    {r['id']}  lang={r.get('lang')}  cat={r.get('cat')}")

# ---- 6. OCR gaps ----
hdr("6. OCR NEAR-EMPTY PAGES (<50 bytes)")
near_empty = sorted(p.name for p in OCR.glob("page_*.txt") if p.stat().st_size < 50)
print(f"  count: {len(near_empty)}")
print(f"  {near_empty}")

# ---- 7. Stratified selection of 100 ----
hdr("7. STRATIFIED SELECTION FOR HUMAN REVIEW (target=100)")


def stratified_pick(pool, target=100, stratify_by=("lang", "diff")):
    """Allocate target across strata proportionally to pool size, then fill."""
    buckets = defaultdict(list)
    for r in pool:
        key = tuple(r.get(k) for k in stratify_by)
        buckets[key].append(r)

    total = len(pool)
    quotas = {}
    for key, items in buckets.items():
        quotas[key] = int(round(len(items) * target / total))

    # adjust to exactly target
    while sum(quotas.values()) < target:
        # give one more to the largest bucket that isn't exhausted
        for key in sorted(buckets, key=lambda k: -len(buckets[k])):
            if quotas[key] < len(buckets[key]):
                quotas[key] += 1
                break
        else:
            break
    while sum(quotas.values()) > target:
        for key in sorted(buckets, key=lambda k: -quotas[k]):
            if quotas[key] > 0:
                quotas[key] -= 1
                break

    selected = []
    for key, q in quotas.items():
        selected.extend(buckets[key][:q])
    return selected, quotas


if len(positives) < 100:
    print(f"  Only {len(positives)} positive candidates — cannot reach 100.")
    selected = positives
else:
    selected, quotas = stratified_pick(positives, target=100, stratify_by=("lang", "diff"))
    print(f"  Selected {len(selected)} rows across strata (lang, diff):")
    for key, q in sorted(quotas.items()):
        print(f"    lang={key[0]:<3} diff={key[1]:<7} -> {q} rows")

print("\n  Selected IDs:")
print("   ", ", ".join(r["id"] for r in selected))

sel_path = Path("data/eval/review_selection.txt")
sel_path.write_text("\n".join(r["id"] for r in selected) + "\n", encoding="utf-8")
print(f"\n  Wrote {sel_path} (one id per line)")

# ---- 8. CSV for Excel ----
with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    f.write("id,lang,cat,diff,answerable,gold_resolved,evidence_found,facts_ok,human_reviewed,selected\n")
    selected_ids = {r["id"] for r in selected}
    for r in rows:
        v = r.get("verification", {})
        f.write(
            f"{r['id']},{r.get('lang','')},{r.get('cat','')},{r.get('diff','')},"
            f"{r.get('answerable')},{v.get('gold_resolved')},{v.get('evidence_found')},"
            f"{facts_ok(r)},{v.get('human_legal_review', False)},"
            f"{r['id'] in selected_ids}\n"
        )
print(f"Wrote {OUT_CSV}")