"""
Junk / non-dialect auditor  (content rules ONLY — never reads model predictions)
================================================================================
Flags eval rows that almost certainly carry NO dialect signal and were labelled
purely by article provenance: bibliographies, ISBN/DOI lines, Latin taxonomy,
English metadata, numeric junk. These inflate the "ceiling" because no model can
assign a dialect to text that isn't in that dialect.

DESIGN RULE (non-negotiable for honesty): flagging depends only on sentence
CONTENT, not on whether any model got it right. A flagged row stays flagged
whether the classifier hit or missed it. This auditor produces CANDIDATES for a
human to eyeball — it is deliberately tuned for precision, not auto-deletion.

    python audit_junk.py --in testset_eval.csv --out flagged.csv

Output: flagged.csv (source,label,rules,sentence) + a console summary.
Then: eyeball flagged.csv, drop the rows you agree with, and RE-SCORE BOTH the
transformer AND the from-scratch baseline on the cleaned eval (cleaning lifts
both — it tests dialect ID more truly, it is not a transformer win by itself).
"""

import argparse
import csv
import re
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="testset_eval.csv")
ap.add_argument("--out", dest="out", default="flagged.csv")
args = ap.parse_args()

# English-only function words (chosen to NOT collide with Italian/dialect forms;
# e.g. 'in'/'a' are excluded because they exist in Italian too).
EN_WORDS = {
    "the", "of", "and", "for", "with", "from", "this", "that", "was", "were",
    "are", "by", "an", "which", "their", "his", "her", "its", "they", "you",
    "have", "has", "been", "about", "between", "during", "against", "warfare",
}
PUBLISHER = re.compile(
    r"\b(press|publish\w*|university|encyclopedia|internet archive|open library|"
    r"openmlol|wikidata|isbn|doi|vol\.|pp\.|2nd|3rd|edition)\b", re.I)
ISBN = re.compile(r"(97[89][\d\-]{9,})|\b\d{1,5}-\d{1,7}-\d{1,7}-[\dxX]\b|\bISBN\b")
DOI = re.compile(r"\b10\.\d{4,9}/\S+")
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
TAXONOMY = re.compile(r"(\bvar\.|\bsubsp\.|\bSp\. ?Pl|\bPl\.:|\bf\. |\bsyn\.)")
URLISH = re.compile(r"https?://|www\.|\.com\b|\.org\b|\(EN\)|\(IT\)")


def en_hits(text):
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return len({t for t in toks if t in EN_WORDS})


def alpha_digit_ratio(text):
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    return alpha, digit, alpha / (alpha + digit + 1)


def rules_for(text):
    fired = []
    if ISBN.search(text) or DOI.search(text):
        fired.append("isbn_doi")
    if TAXONOMY.search(text):
        fired.append("latin_taxonomy")
    if URLISH.search(text):
        fired.append("url_metadata")
    if YEAR.search(text) and PUBLISHER.search(text):
        fired.append("citation_shape")
    if en_hits(text) >= 3:
        fired.append("english_text")
    alpha, digit, ratio = alpha_digit_ratio(text)
    if digit >= 6 or (alpha >= 3 and ratio < 0.55):
        fired.append("numeric_junk")
    # very short fragment with a trailing-initial citation tail e.g. 'La chiesa di S.'
    if len(text.split()) <= 5 and re.search(r"\b[A-Z]\.\s*$", text.strip()):
        fired.append("citation_fragment")
    return fired


def read_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    has_src = len(rows[0]) >= 3
    return [(r[0], r[1], r[2] if has_src and len(r) >= 3 else "")
            for r in rows[1:] if len(r) >= 2]


data = read_csv(args.inp)
flagged = []
for sent, label, src in data:
    fired = rules_for(sent)
    if fired:
        flagged.append((src, label, ";".join(fired), sent))

with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source", "label", "rules", "sentence"])
    w.writerows(flagged)

# ---- summary ----
total = len(data)
n = len(flagged)
print(f"Scanned {total} rows | flagged {n} ({n/total:.1%}) -> {args.out}")

by_rule = Counter(r for _, _, rules, _ in flagged for r in rules.split(";"))
by_label = Counter(lab for _, lab, _, _ in flagged)
by_src = Counter(src for src, _, _, _ in flagged)

print("\n  by rule (a row may fire several):")
for r, c in by_rule.most_common():
    print(f"    {c:>4}  {r}")
print("\n  by label (how much each class shrinks if all confirmed):")
labels_total = Counter(lab for _, lab, _ in data)
for lab in sorted(labels_total):
    print(f"    {lab:>10}: {by_label.get(lab,0):>3} flagged / {labels_total[lab]:>4} total")
print("\n  top sources of flags:")
for s, c in by_src.most_common(10):
    print(f"    {c:>4}  {s}")
print("\nNEXT: eyeball flagged.csv. It is candidates, not a delete list — keep any "
      "real dialect text. Then re-score BOTH models on the cleaned eval.")
