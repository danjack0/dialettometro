"""
split_product_v2.py — rebuild the product split with a LEAK-FREE `other` class
==============================================================================
Fixes issue D (VERIFIED_FACTS.md): in the original product_config, every one of
the 500 test-`other` rows was also in train-`other`, because split_configs.py
appended `other_data.csv` whole to train and `other_eval.csv` (a subset of it)
whole to test. The `other` class therefore had no real held-out set.

This builder writes NEW files (data/configs/product_v2_*) and never touches the
existing product_config_* / itdi_parity_* artifacts.

What it does
------------
1. Carries over every NON-`other` row from the existing product_config split
   VERBATIM (same train/test membership), so the only thing that changes versus
   product_config is the `other` class — any macro-F1 movement is attributable to
   the fix, not to a reshuffled dialect split.
2. Re-partitions the full `other` pool DISJOINTLY, grouped by Tatoeba language
   subset, via per_label_group_split keyed on `source` (= "tatoeba:<lang>"). No
   language straddles the split: test-`other` languages are held out of training
   entirely, which is the honest way to measure a rejection class.
3. Drops from the test side any row whose sentence still appears in train (there
   is a single coincidental `sicilian` duplicate carried over from
   product_config), then asserts zero leakage with split_utils.assert_no_leak.

    python scripts/split_product_v2.py
"""

import csv
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_utils import per_label_group_split, assert_no_leak, SPLIT_SEED

PROD_TRAIN = "data/configs/product_config_train.csv"
PROD_TEST = "data/configs/product_config_test.csv"
OTHER_POOL = "data/raw/other_data.csv"          # full distinct `other` pool (11,798; eval ⊂ this)
OUT_TRAIN = "data/configs/product_v2_train.csv"
OUT_TEST = "data/configs/product_v2_test.csv"
HEADER = ["sentence", "label", "source"]
TEST_SIZE = 0.2


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [(r[0], r[1], r[2] if len(r) >= 3 else "") for r in rows[1:] if len(r) >= 2]


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def counts(rows):
    return Counter(r[1] for r in rows)


def main():
    prod_train = read_rows(PROD_TRAIN)
    prod_test = read_rows(PROD_TEST)

    # 1. non-`other` rows carried over verbatim
    nonother_train = [r for r in prod_train if r[1] != "other"]
    nonother_test = [r for r in prod_test if r[1] != "other"]

    # 2. disjoint, language-grouped split of the full `other` pool
    other_pool = read_rows(OTHER_POOL)
    assert all(r[1] == "other" for r in other_pool), "OTHER_POOL must be all label=other"
    other_train, other_test = per_label_group_split(other_pool, test_size=TEST_SIZE,
                                                    seed=SPLIT_SEED, verbose=True)
    tr_langs = sorted({r[2] for r in other_train})
    te_langs = sorted({r[2] for r in other_test})
    straddle = set(tr_langs) & set(te_langs)
    print(f"\n`other` split: {len(other_train)} train rows / {len(other_test)} test rows")
    print(f"  train languages ({len(tr_langs)}): {[l.split(':')[1] for l in tr_langs]}")
    print(f"  test  languages ({len(te_langs)}): {[l.split(':')[1] for l in te_langs]}")
    assert not straddle, f"language(s) straddling the split: {straddle}"
    print(f"  languages straddling split: {len(straddle)}  (must be 0)")

    v2_train = nonother_train + other_train
    v2_test = nonother_test + other_test

    # 3. drop any test row whose sentence is also in train (the lone sicilian dup)
    train_sent = {r[0] for r in v2_train}
    before = len(v2_test)
    dropped = [r for r in v2_test if r[0] in train_sent]
    v2_test = [r for r in v2_test if r[0] not in train_sent]
    if dropped:
        print(f"\ndropped {len(dropped)} leaked test row(s) carried over from product_config: "
              f"{Counter(r[1] for r in dropped)}")

    # final guard — must pass
    assert_no_leak(v2_train, v2_test, stage="product_v2")
    print("assert_no_leak: PASS (zero train/test sentence overlap in any class)")

    write_rows(OUT_TRAIN, v2_train)
    write_rows(OUT_TEST, v2_test)

    tr_c, te_c = counts(v2_train), counts(v2_test)
    print(f"\nwrote {OUT_TRAIN} ({len(v2_train)}) and {OUT_TEST} ({len(v2_test)})")
    print(f"{'class':>12}  {'train':>6}  {'test':>5}  {'%test':>6}")
    for c in sorted(set(tr_c) | set(te_c)):
        t, e = tr_c.get(c, 0), te_c.get(c, 0)
        pct = e / (t + e) * 100 if (t + e) else 0
        print(f"{c:>12}  {t:>6}  {e:>5}  {pct:5.1f}%")


if __name__ == "__main__":
    main()
