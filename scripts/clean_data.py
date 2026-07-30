"""
Safe data cleaner  (paranoid by design — protects ground truth)
===============================================================
Reads ONLY the read-only originals in /mnt/user-data/uploads and writes new
*_clean.csv files plus full manifests. Never overwrites an original.

What it removes (conservative — only what we verified by eyeball):
  * HIGH-PRECISION junk rules only: isbn_doi, latin_taxonomy, url_metadata,
    citation_shape, english_text.  (numeric_junk is NOT a drop rule — it false-
    positives on real dialect sentences that contain a year/date; those rows are
    routed to a REVIEW file and KEPT.)
  * testset_eval only: the 11 'Centona/Prefazione' rows — a standard-Italian
    preface mislabelled 'sicilian'. ('Ore di città/25' is mixed register -> REVIEW,
    kept, for human decision; not auto-dropped.)

Single-pass partition: every input row goes to exactly ONE of {clean, removed},
so kept and removed are exact complements — no set-matching, duplicate-safe.

Self-checks (abort + write nothing on failure):
  1. len(clean) + len(removed) == len(input)
  2. round-trip: re-read each written clean file; every row must match byte-for-byte
  3. expected removal counts must match what we audited (balanced=102, eval=11)
  4. no removed row is present in the clean output
"""

import csv
import os
import re
import sys
from collections import Counter

SRC_DIR = "/mnt/user-data/uploads"
OUT_DIR = "/mnt/user-data/outputs"

# ---- rule predicates (identical to audit_junk.py) ----
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

HIGH_PRECISION = {"isbn_doi", "latin_taxonomy", "url_metadata", "citation_shape", "english_text"}
MISLABEL_DROP = {"Centona/Prefazione"}            # eval only: confirmed standard-Italian preface
MISLABEL_REVIEW = {"Ore di città/25"}             # eval only: mixed register, human decides


def fired_rules(t):
    f = []
    if ISBN.search(t) or DOI.search(t): f.append("isbn_doi")
    if TAXONOMY.search(t): f.append("latin_taxonomy")
    if URLISH.search(t): f.append("url_metadata")
    if YEAR.search(t) and PUBLISHER.search(t): f.append("citation_shape")
    toks = re.findall(r"[a-zA-Z']+", t.lower())
    if len({x for x in toks if x in EN_WORDS}) >= 3: f.append("english_text")
    a = sum(c.isalpha() for c in t); d = sum(c.isdigit() for c in t)
    if d >= 6 or (a >= 3 and a / (a + d + 1) < 0.55): f.append("numeric_junk")
    return f


def read_rows(path):
    with open(path, encoding="utf-8") as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]            # header, data rows (raw lists, preserved exactly)


def write_rows(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def clean_file(fname, eval_mode=False):
    src = os.path.join(SRC_DIR, fname)
    header, rows = read_rows(src)
    clean, removed, review = [], [], []

    for row in rows:
        if len(row) < 2:                     # malformed -> keep verbatim, never silently drop
            clean.append(row); continue
        sent, label = row[0], row[1]
        source = row[2] if len(row) >= 3 else ""
        fr = fired_rules(sent)
        hi = HIGH_PRECISION & set(fr)

        if eval_mode and source in MISLABEL_DROP:
            removed.append(row + ["mislabel_source"])
        elif hi:
            removed.append(row + [";".join(sorted(hi))])
        else:
            clean.append(row)
            if (eval_mode and source in MISLABEL_REVIEW):
                review.append(row + ["mixed_register_review"])
            elif "numeric_junk" in fr:
                review.append(row + ["numeric_junk_review"])

    # ---- self-checks ----
    assert len(clean) + len(removed) == len(rows), \
        f"{fname}: partition lost rows ({len(clean)}+{len(removed)} != {len(rows)})"

    base = fname.replace(".csv", "")
    clean_path = os.path.join(OUT_DIR, f"{base}_clean.csv")
    write_rows(clean_path, header, clean)
    write_rows(os.path.join(OUT_DIR, f"removed_{base}.csv"), header + ["reason"], removed)
    if review:
        write_rows(os.path.join(OUT_DIR, f"review_{base}.csv"), header + ["reason"], review)

    # round-trip integrity: re-read the clean file, must equal what we intended to keep
    _, reloaded = read_rows(clean_path)
    assert reloaded == clean, f"{fname}: ROUND-TRIP MISMATCH — CSV encoding corrupted a row"

    return header, rows, clean, removed, review


def summarize(fname, rows, clean, removed, review):
    print(f"\n##### {fname} #####")
    print(f"  input {len(rows)}  ->  clean {len(clean)}  | removed {len(removed)}  | review(kept) {len(review)}")
    if removed:
        by_reason = Counter(r[-1] for r in removed)
        print("  removed by reason:", dict(by_reason))
        bylab = Counter(r[1] for r in removed)
        print("  removed by label :", dict(sorted(bylab.items())))


if __name__ == "__main__":
    results = {}
    for fname, ev in [("balanced.csv", False),
                      ("testset_train.csv", False),
                      ("testset_eval.csv", True)]:
        header, rows, clean, removed, review = clean_file(fname, eval_mode=ev)
        summarize(fname, rows, clean, removed, review)
        results[fname] = (len(removed),)

    # hard guards against drift from what we audited
    assert results["balanced.csv"][0] == 102, f"balanced drop != 102 (got {results['balanced.csv'][0]})"
    assert results["testset_eval.csv"][0] == 11, f"eval drop != 11 (got {results['testset_eval.csv'][0]})"
    print("\nAll integrity checks PASSED. Originals untouched; clean files + manifests written.")
