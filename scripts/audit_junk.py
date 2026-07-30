"""
Junk / non-dialect auditor  (content rules ONLY — never reads model predictions)
================================================================================
Two independent passes, both blind to model output:

  PASS 1 — NO-SIGNAL JUNK  (deletion candidates)
    Rows that almost certainly carry NO dialect signal and were labelled purely
    by article provenance: bibliographies, ISBN/DOI lines, Latin taxonomy,
    English metadata, numeric junk. No model can assign a dialect to text that
    isn't in that dialect, so these inflate the apparent "ceiling".

  PASS 2 — TEMPLATE / NEAR-DUPLICATE BOILERPLATE  (reduction candidates)
    Bot-generated stub sentences ("X is a comune in the province of Y ...") ARE
    real dialect text, so Pass 1 rightly leaves them alone — but they recur with
    only a proper-noun slot changing across hundreds of distinct articles. That
    causes n-gram overfitting to template tokens and near-duplicate leakage
    across the article-level split (different source, near-identical sentence).
    These are NOT deletion candidates: you dedup / downsample them, you do not
    wipe a class's whole template inventory. Added for the ITDI expansion, where
    small wikis (esp. Piedmontese) are heavily bot-inflated and raw article count
    massively overstates usable text.

DESIGN RULE (non-negotiable for honesty): flagging depends only on sentence
CONTENT (and, for Pass 2, on how many distinct SOURCES share a skeleton) — never
on whether any model got the row right. A flagged row stays flagged whether the
classifier hit or missed it. This auditor produces CANDIDATES for a human to
eyeball; it is deliberately tuned for precision, not auto-deletion.

    python audit_junk.py --in balanced_11class.csv --out flagged.csv

Output: flagged.csv (source,label,rules,sentence) + a console summary.
Then: eyeball flagged.csv. For no-signal rules, drop the rows you agree with. For
template/near-duplicate rows, dedup or downsample per class. Finally RE-SCORE
BOTH the transformer AND the from-scratch baseline on the cleaned data (cleaning
lifts both — it tests dialect ID more truly, it is not a transformer win).
"""

import argparse
import csv
import re
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="testset_eval.csv")
ap.add_argument("--out", dest="out", default="flagged.csv")
ap.add_argument("--template-min-sources", type=int, default=8,
                help="a sentence skeleton shared by >= this many DISTINCT sources "
                     "is flagged as template boilerplate (default 8)")
ap.add_argument("--dup-min-count", type=int, default=3,
                help="a normalised sentence appearing >= this many times is "
                     "flagged as a near-duplicate (default 3)")
ap.add_argument("--no-template", action="store_true",
                help="skip Pass 2 (template/near-duplicate detection)")
ap.add_argument("--numeric-min-digits", type=int, default=10,
                 help="minimum digit count before a row can be flagged as a "
                      "pure numeric-junk row (default 10)")
ap.add_argument("--numeric-max-ratio", type=float, default=0.35,
                 help="alpha/(alpha+digit+1) ceiling for numeric_junk — below "
                      "this, digits are judged to DOMINATE the sentence rather "
                      "than merely appear in normal dated/statistical prose "
                      "(default 0.35)")
args = ap.parse_args()

# --- Pass 1: no-signal rules --------------------------------------------------
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
    # digit-dominant rows only: a sentence with a few dates/populations mixed
    # into real prose (very common in comune/biography stub articles) must NOT
    # fire here — only rows where digits swamp the actual text should.
    if digit >= args.numeric_min_digits and ratio < args.numeric_max_ratio:
        fired.append("numeric_junk")
    # very short fragment with a trailing-initial citation tail e.g. 'La chiesa di S.'
    if len(text.split()) <= 5 and re.search(r"\b[A-Z]\.\s*$", text.strip()):
        fired.append("citation_fragment")
    return fired


# --- Pass 2: template / near-duplicate detection ------------------------------
# skeleton(): collapse the variable slots of a template sentence (proper-noun
# slots and numbers) to a canonical frame, so bot geo/bio stubs that differ only
# by place name map to ONE key regardless of dialect. Language-agnostic on
# purpose — it keys on structure, not vocabulary.
_DIGITS = re.compile(r"\d+")
_CAP = re.compile(r"\b[A-ZÀ-Þ][\w'’À-ÿ]*")   # Capitalised token = proper-noun slot
_PUNCT = re.compile(r"[^\w\s·]", re.UNICODE)
_WS = re.compile(r"\s+")


def skeleton(text):
    s = _DIGITS.sub("0", text)
    s = _CAP.sub("·", s)          # blank out proper-noun slots
    s = _PUNCT.sub(" ", s)        # 'comune,' == 'comune'
    s = _WS.sub(" ", s).strip().lower()
    return s


def normalise(text):
    # for exact near-duplicate detection: punctuation/whitespace/case-insensitive
    s = _PUNCT.sub(" ", text)
    return _WS.sub(" ", s).strip().lower()


def template_flags(rows, min_sources, dup_min):
    """rows: list of (sent, label, src). Returns dict idx -> set(rule)."""
    skel_sources = defaultdict(set)   # skeleton -> {distinct sources}
    skel_rows = defaultdict(list)     # skeleton -> [row idx]
    norm_rows = defaultdict(list)     # normalised text -> [row idx]
    for i, (sent, _lab, src) in enumerate(rows):
        sk = skeleton(sent)
        # ignore trivially short skeletons (they collapse to noise)
        if len(sk.split()) >= 4:
            skel_sources[sk].add(src or i)
            skel_rows[sk].append(i)
        norm_rows[normalise(sent)].append(i)

    flags = defaultdict(set)
    for sk, srcs in skel_sources.items():
        if len(srcs) >= min_sources:
            for i in skel_rows[sk]:
                flags[i].add("template_boilerplate")
    for _norm, idxs in norm_rows.items():
        if len(idxs) >= dup_min:
            for i in idxs:
                flags[i].add("near_duplicate")
    return flags


# --- IO -----------------------------------------------------------------------
def read_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    has_src = len(rows[0]) >= 3
    return [(r[0], r[1], r[2] if has_src and len(r) >= 3 else "")
            for r in rows[1:] if len(r) >= 2]


data = read_csv(args.inp)

# Pass 1 (per-row, content-only)
row_rules = [rules_for(sent) for sent, _, _ in data]

# Pass 2 (corpus-wide, content + source-diversity only)
tmpl = defaultdict(set)
if not args.no_template:
    tmpl = template_flags(data, args.template_min_sources, args.dup_min_count)

flagged = []
for i, (sent, label, src) in enumerate(data):
    fired = list(row_rules[i]) + sorted(tmpl.get(i, ()))
    if fired:
        flagged.append((src, label, ";".join(fired), sent))

with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source", "label", "rules", "sentence"])
    w.writerows(flagged)

# ---- summary ----
NO_SIGNAL = {"isbn_doi", "latin_taxonomy", "url_metadata", "citation_shape",
             "english_text", "numeric_junk", "citation_fragment"}
REDUCTION = {"template_boilerplate", "near_duplicate"}

total = len(data)
n = len(flagged)
print(f"Scanned {total} rows | flagged {n} ({n/total:.1%}) -> {args.out}")

by_rule = Counter(r for _, _, rules, _ in flagged for r in rules.split(";"))
by_label = Counter(lab for _, lab, _, _ in flagged)
by_src = Counter(src for src, _, _, _ in flagged)

print("\n  by rule (a row may fire several):")
for r, c in by_rule.most_common():
    kind = "delete?" if r in NO_SIGNAL else ("reduce?" if r in REDUCTION else "")
    print(f"    {c:>5}  {r:<20} {kind}")

# split no-signal vs reduction per class — the two need different treatment
print("\n  per class  [no-signal = delete candidates | template/dup = reduce candidates]:")
labels_total = Counter(lab for _, lab, _ in data)
ns_by_label = Counter()
rd_by_label = Counter()
for src, lab, rules, sent in flagged:
    rs = set(rules.split(";"))
    if rs & NO_SIGNAL:
        ns_by_label[lab] += 1
    if rs & REDUCTION:
        rd_by_label[lab] += 1
print(f"    {'class':>16}  {'total':>7}  {'no-signal':>10}  {'template/dup':>13}")
for lab in sorted(labels_total):
    t = labels_total[lab]
    ns = ns_by_label.get(lab, 0)
    rd = rd_by_label.get(lab, 0)
    print(f"    {lab:>16}  {t:>7}  {ns:>6} {ns/t:>5.1%}  {rd:>7} {rd/t:>5.1%}")

print("\n  top sources of flags:")
for s, c in by_src.most_common(10):
    print(f"    {c:>5}  {s}")

print("\nNEXT: eyeball flagged.csv — it is candidates, not a delete list.")
print("  • no-signal rows  -> drop the ones you agree carry no dialect signal.")
print("  • template/dup rows -> dedup or downsample per class; DO NOT delete a")
print("    class's whole template inventory (that text is real dialect).")
print("  Then re-score BOTH models on the cleaned data.")