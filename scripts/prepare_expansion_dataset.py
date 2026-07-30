"""
prepare_expansion_dataset.py
=============================
Builds the balanced 13-class product-config training set from combined_raw.csv
(the full raw scrape across all wikis + Tatoeba). Uses the SAME content-only
rules as audit_junk.py, so a row's fate depends only on its own text and,
for Pass 2, on how many distinct sources share its skeleton — never on any
model's prediction and never on its label being "important" or not.

Single-pass partition: every input row lands in exactly ONE of
{clean, removed_stage1, reduced_stage2, capped_stage3}. Counts must sum to the
input length — checked below, abort + write nothing on failure.

  STAGE 1 — no-signal removal (isbn_doi, latin_taxonomy, url_metadata,
            citation_shape, english_text, numeric_junk). Uses the CORRECTED
            digit-dominance numeric_junk rule (digit>=N and ratio<R), not the
            old digit>=6 rule, which false-positived heavily on ordinary dated/
            statistical prose (comune population figures, biography dates) —
            verified on 25 real sampled rows during the audit, 0 were junk.

  STAGE 2 — template/near-duplicate reduction. A sentence skeleton shared by
            >= --template-min-sources DISTINCT source articles is bot
            boilerplate carrying repeated structure, not repeated dialect
            signal; keep only --template-keep copies of it, chosen to maximize
            how many distinct sources are represented among the survivors
            (so we don't just keep the first N alphabetically). Exact verbatim
            near-duplicates (>= --dup-min-count copies) collapse to one copy.
            This is a REDUCTION, never a full deletion of a class's template
            inventory — the surviving copies are real dialect text.

  STAGE 3 — per-class capping. Classes over --class-cap are downsampled to it,
            capped at --per-source-cap rows from any single source article so
            one prolific article can't dominate the surviving sample. Classes
            already under --class-cap (the thin ones: Ladin, Emilian,
            Piedmontese, and the original five) are left completely untouched
            — no oversampling is performed here. Remaining imbalance after
            capping is meant to be handled by class-weighted loss at train
            time, not further data manipulation.

Usage:
    python prepare_expansion_dataset.py --in combined_raw.csv \
        --out balanced_13class.csv --class-cap 4000 --per-source-cap 15
"""

import argparse
import csv
import random
import re
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
ap.add_argument("--removed-out", default="removed_expansion.csv")
ap.add_argument("--template-min-sources", type=int, default=8)
ap.add_argument("--template-keep", type=int, default=3,
                 help="max surviving copies per flagged template skeleton (default 3)")
ap.add_argument("--dup-min-count", type=int, default=3)
ap.add_argument("--class-cap", type=int, default=4000,
                 help="ceiling per class after stages 1-2; classes already "
                      "under this are left untouched (default 4000)")
ap.add_argument("--per-source-cap", type=int, default=15,
                 help="max rows kept from any single source article during "
                      "stage-3 capping, to preserve source diversity (default 15)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)

# ---- Stage 1 rules: identical to audit_junk.py's corrected version ----
EN_WORDS = {"the","of","and","for","with","from","this","that","was","were","are",
            "by","an","which","their","his","her","its","they","you","have","has",
            "been","about","between","during","against","warfare"}
PUBLISHER = re.compile(r"\b(press|publish\w*|university|encyclopedia|internet archive|"
                       r"open library|openmlol|wikidata|isbn|doi|vol\.|pp\.|2nd|3rd|edition)\b", re.I)
ISBN = re.compile(r"(97[89][\d\-]{9,})|\b\d{1,5}-\d{1,7}-\d{1,7}-[\dxX]\b|\bISBN\b")
DOI = re.compile(r"\b10\.\d{4,9}/\S+")
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
TAXONOMY = re.compile(r"(\bvar\.|\bsubsp\.|\bSp\. ?Pl|\bPl\.:|\bf\. |\bsyn\.)")
URLISH = re.compile(r"https?://|www\.|\.com\b|\.org\b|\(EN\)|\(IT\)")
NUMERIC_MIN_DIGITS, NUMERIC_MAX_RATIO = 10, 0.35


def stage1_rules(text):
    fired = []
    if ISBN.search(text) or DOI.search(text): fired.append("isbn_doi")
    if TAXONOMY.search(text): fired.append("latin_taxonomy")
    if URLISH.search(text): fired.append("url_metadata")
    if YEAR.search(text) and PUBLISHER.search(text): fired.append("citation_shape")
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    if len({t for t in toks if t in EN_WORDS}) >= 3: fired.append("english_text")
    alpha = sum(c.isalpha() for c in text); digit = sum(c.isdigit() for c in text)
    ratio = alpha / (alpha + digit + 1)
    if digit >= NUMERIC_MIN_DIGITS and ratio < NUMERIC_MAX_RATIO:
        fired.append("numeric_junk")
    if len(text.split()) <= 5 and re.search(r"\b[A-Z]\.\s*$", text.strip()):
        fired.append("citation_fragment")
    return fired


# ---- Stage 2: skeleton/normalise, identical to audit_junk.py ----
_DIGITS = re.compile(r"\d+")
_CAP = re.compile(r"\b[A-ZÀ-Þ][\w'’À-ÿ]*")
_PUNCT = re.compile(r"[^\w\s·]", re.UNICODE)
_WS = re.compile(r"\s+")


def skeleton(text):
    s = _DIGITS.sub("0", text)
    s = _CAP.sub("·", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip().lower()


def normalise(text):
    s = _PUNCT.sub(" ", text)
    return _WS.sub(" ", s).strip().lower()


# ---- IO ----
def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    return header, rows[1:]


header, rows = read_csv(args.inp)
n_total = len(rows)

clean, removed1, reduced2, capped3 = [], [], [], []

# --- Stage 1 ---
survivors1 = []
for row in rows:
    sent = row[0]
    fired = stage1_rules(sent)
    if fired:
        removed1.append(row + [";".join(fired)])
    else:
        survivors1.append(row)

# --- Stage 2: template skeleton + near-dup grouping, computed on survivors1 only ---
skel_groups = defaultdict(list)   # skeleton -> [row indices into survivors1]
norm_groups = defaultdict(list)   # normalised text -> [row indices]
for i, row in enumerate(survivors1):
    sent, src = row[0], (row[2] if len(row) >= 3 else "")
    sk = skeleton(sent)
    if len(sk.split()) >= 4:
        skel_groups[sk].append(i)
    norm_groups[normalise(sent)].append(i)

drop_idx = set()   # indices into survivors1 to reduce out
reduce_reason = {}

for sk, idxs in skel_groups.items():
    sources = {survivors1[i][2] if len(survivors1[i]) >= 3 else i for i in idxs}
    if len(sources) >= args.template_min_sources:
        # keep args.template_keep copies, preferring source diversity: sort by
        # source then round-robin-pick to spread survivors across many sources
        by_source = defaultdict(list)
        for i in idxs:
            s = survivors1[i][2] if len(survivors1[i]) >= 3 else i
            by_source[s].append(i)
        keep = []
        pools = list(by_source.values())
        random.shuffle(pools)
        pi = 0
        while len(keep) < args.template_keep and any(pools):
            pool = pools[pi % len(pools)]
            if pool:
                keep.append(pool.pop())
            pi += 1
            if pi > 10000:
                break
        keep_set = set(keep)
        for i in idxs:
            if i not in keep_set:
                drop_idx.add(i)
                reduce_reason[i] = "template_boilerplate"

for norm, idxs in norm_groups.items():
    if len(idxs) >= args.dup_min_count:
        keep_one = idxs[0]
        for i in idxs[1:]:
            if i not in drop_idx:
                drop_idx.add(i)
            reduce_reason.setdefault(i, "near_duplicate")

survivors2 = []
for i, row in enumerate(survivors1):
    if i in drop_idx:
        reduced2.append(row + [reduce_reason.get(i, "template_boilerplate")])
    else:
        survivors2.append(row)

# --- Stage 3: per-class cap with per-source diversity cap ---
by_class = defaultdict(list)
for row in survivors2:
    by_class[row[1]].append(row)

final_clean = []
for label, class_rows in by_class.items():
    if len(class_rows) <= args.class_cap:
        final_clean.extend(class_rows)
        continue
    # source-diversity-aware sample: cap any single source at --per-source-cap,
    # then randomly fill remaining budget from the rest
    by_source = defaultdict(list)
    for row in class_rows:
        src = row[2] if len(row) >= 3 else ""
        by_source[src].append(row)
    capped_pool, overflow = [], []
    for src, rs in by_source.items():
        random.shuffle(rs)
        capped_pool.extend(rs[:args.per_source_cap])
        overflow.extend(rs[args.per_source_cap:])
    random.shuffle(capped_pool)
    random.shuffle(overflow)
    if len(capped_pool) >= args.class_cap:
        keep = capped_pool[:args.class_cap]
        drop = capped_pool[args.class_cap:] + overflow
    else:
        need = args.class_cap - len(capped_pool)
        keep = capped_pool + overflow[:need]
        drop = overflow[need:]
    final_clean.extend(keep)
    for row in drop:
        capped3.append(row + ["class_cap_downsample"])

# ---- integrity checks ----
assert len(final_clean) + len(removed1) + len(reduced2) + len(capped3) == n_total, \
    f"partition lost rows: {len(final_clean)}+{len(removed1)}+{len(reduced2)}+{len(capped3)} != {n_total}"

with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(final_clean)

with open(args.removed_out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header + ["reason"])
    w.writerows(removed1 + reduced2 + capped3)

# round-trip check
with open(args.out, encoding="utf-8") as f:
    reloaded = list(csv.reader(f))[1:]
assert reloaded == final_clean, "ROUND-TRIP MISMATCH — CSV encoding corrupted a row"

# ---- summary ----
print(f"Input {n_total} rows")
print(f"  stage1 no-signal removed : {len(removed1)}")
print(f"  stage2 template/dup reduced : {len(reduced2)}")
print(f"  stage3 class-cap downsampled : {len(capped3)}")
print(f"  FINAL clean dataset : {len(final_clean)} -> {args.out}")
print(f"  full removal manifest -> {args.removed_out}")

print("\nper-class before -> after:")
before_counts = Counter(row[1] for row in rows)
after_counts = Counter(row[1] for row in final_clean)
for label in sorted(before_counts):
    b, a = before_counts[label], after_counts.get(label, 0)
    print(f"  {label:>16}  {b:>7} -> {a:>6}")

print("\nAll integrity checks PASSED.")
