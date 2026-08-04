"""
split_configs.py
=================
Builds train/test splits for BOTH configs from one cleaned dataset, using
GroupShuffleSplit keyed on `source` (article) so no article's sentences ever
appear in both train and test.

IMPORTANT: the split is done PER LABEL, not once globally. A single global
split targets an 80/20 ratio by GROUP COUNT, not row count — if one class's
rows are concentrated in a few large-article groups, luck in which groups
land in test can badly skew that one class's row ratio even while every other
class looks fine (this happened to Ligurian: 1854 test vs 568 train on the
first global-split attempt). Splitting each label's own groups independently
keeps every class close to the target ratio by rows.

`other` is handled separately: its `source` field is a language tag
(tatoeba:eng, tatoeba:lat), not an article, so grouping by it would be
meaningless. Instead `--other-train`/`--other-eval` are appended directly:
all of --other-train goes to the product-config train split, all of
--other-eval goes to the product-config test split (matching their filenames
— other_eval.csv is already a curated held-out set, not raw pool to resplit).
`other` is excluded from the ITDI-parity config entirely (matches benchmark).

Usage:
    python split_configs.py --in balanced_13class.csv \
        --itdi-out itdi_parity --product-out product_config \
        --other-train other_data.csv --other-eval other_eval.csv \
        --test-size 0.2
"""

import argparse
import csv
from collections import Counter

from split_utils import group_key, per_label_group_split, assert_no_leak

ITDI_11 = {"piedmontese", "venetian", "sicilian", "neapolitan", "emilian",
           "tarantino", "sardinian", "ligurian", "friulian", "ladin", "lombard"}

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--itdi-out", required=True)
ap.add_argument("--product-out", required=True)
ap.add_argument("--other-pool", default=None,
                help="single `other` CSV (e.g. other_data.csv). PREFERRED: it is "
                     "split DISJOINTLY here, grouped by Tatoeba language (the `source` "
                     "column), so no language and no sentence straddles train/test.")
ap.add_argument("--other-train", default=None,
                help="LEGACY/UNSAFE: other rows appended whole to train. If the "
                     "matching --other-eval is a subset of this file, every eval row "
                     "also lands in train — the exact bug that made the original "
                     "product_config's `other` class un-held-out. Prefer --other-pool. "
                     "The final assert_no_leak below will refuse to write a leaked split.")
ap.add_argument("--other-eval", default=None,
                help="LEGACY/UNSAFE: other rows appended whole to test. See --other-train.")
ap.add_argument("--test-size", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


# --in is balanced_13class.csv, which despite its name holds 12 labels: the 11
# dialects + `standard`. The 13th class, `other`, is NOT in this file — it is
# appended below from --other-train/--other-eval when the product split is built
# (the ITDI-parity split drops `standard` too, leaving 11).
header, data = read_rows(args.inp)


def leak_check(train_rows, test_rows, label):
    # qualify by (row_label, source) — identical article TITLES legitimately
    # recur across different dialect wikis (e.g. "Torino" exists as a separate,
    # unrelated article on both the Piedmontese and Sardinian wikis). Checking
    # bare source strings across labels produces false-positive "leakage" for
    # these coincidental title collisions; real leakage is the same label's
    # same source appearing on both sides.
    train_srcs = {(r[1], group_key(r, i)) for i, r in enumerate(train_rows)}
    test_srcs = {(r[1], group_key(r, i)) for i, r in enumerate(test_rows)}
    overlap = train_srcs & test_srcs
    assert not overlap, f"{label}: {len(overlap)} (label,source) pairs leaked across train/test"


def report(train_rows, test_rows, label):
    print(f"\n{label}: {len(train_rows)+len(test_rows)} rows -> "
          f"train {len(train_rows)} / test {len(test_rows)}")
    tr_c, te_c = Counter(r[1] for r in train_rows), Counter(r[1] for r in test_rows)
    for lab in sorted(set(tr_c) | set(te_c)):
        t, e = tr_c.get(lab, 0), te_c.get(lab, 0)
        pct = e / (t + e) * 100 if (t + e) else 0
        flag = "  <-- CHECK" if pct > 35 or pct < 8 else ""
        print(f"    {lab:>16}  train {t:>6}  test {e:>5}  ({pct:4.1f}% test){flag}")


def write(rows_out, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows_out)


# ---- ITDI-parity 11-class (no 'other', no 'standard') ----
itdi_rows = [r for r in data if r[1] in ITDI_11]
missing = ITDI_11 - {r[1] for r in itdi_rows}
if missing:
    print(f"WARNING: ITDI-parity config missing classes entirely: {sorted(missing)}")
itdi_train, itdi_test = per_label_group_split(itdi_rows, args.test_size, args.seed)
leak_check(itdi_train, itdi_test, "ITDI-parity")
write(itdi_train, f"{args.itdi_out}_train.csv")
write(itdi_test, f"{args.itdi_out}_test.csv")
report(itdi_train, itdi_test, "ITDI-parity (11-class)")

# ---- Product config: all classes present, plus 'other' ----
prod_train, prod_test = per_label_group_split(data, args.test_size, args.seed)
leak_check(prod_train, prod_test, "Product config (dialect classes)")

# PREFERRED: split a single `other` pool the same source-aware way as every other
# class. For `other`, the `source` column is the Tatoeba language tag, so grouping
# by it keeps whole languages together — no language (and so no sentence) can
# straddle train and test.
if args.other_pool:
    _, other_rows = read_rows(args.other_pool)
    other_rows = [tuple(r[:3]) for r in other_rows]
    other_tr, other_te = per_label_group_split(other_rows, args.test_size, args.seed)
    prod_train.extend(other_tr)
    prod_test.extend(other_te)

# LEGACY path: append pre-split files whole. Kept for compatibility, but the
# assert_no_leak below is what stops the historical bug (eval ⊂ train) from ever
# being written again — it inspects the FINAL rows, AFTER these appends, which the
# old (label, source) leak_check never did.
if args.other_train:
    _, other_tr_rows = read_rows(args.other_train)
    prod_train.extend(tuple(r[:3]) for r in other_tr_rows)
if args.other_eval:
    _, other_te_rows = read_rows(args.other_eval)
    prod_test.extend(tuple(r[:3]) for r in other_te_rows)

# ROOT-CAUSE GUARD: sentence-level, every class, on the post-append rows. This is
# the check the original pipeline lacked — leak_check() above ran only on the
# dialect split, before `other` was appended, so the leaked `other` eval set
# sailed through.
assert_no_leak(prod_train, prod_test, stage="product config (final)")

write(prod_train, f"{args.product_out}_train.csv")
write(prod_test, f"{args.product_out}_test.csv")
report(prod_train, prod_test, "Product config (all classes)")

present_labels = sorted({r[1] for r in prod_train} | {r[1] for r in prod_test})
print(f"\nProduct config classes present ({len(present_labels)}): {present_labels}")
if "other" not in present_labels:
    print("NOTE: 'other' still missing — pass --other-train/--other-eval.")