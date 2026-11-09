"""
Tests for scripts/split_utils.py: the leak check and the row-ratio,
article-level split. These are the two bugs documented in the README
("Fixed: the Product `other` class" and "Fixed: GroupShuffleSplit skewed
per-class test ratios"); the tests pin the fixes in place.

Run: pytest -q
"""

import csv
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from split_utils import assert_no_leak, per_label_group_split, split_indices  # noqa: E402


# ---------- helpers ----------

def make_rows(label, article_sizes, prefix=None):
    """One label's rows: article i contributes article_sizes[i] sentences."""
    prefix = prefix or label
    rows = []
    for a, n in enumerate(article_sizes):
        for s in range(n):
            rows.append((f"{prefix} art{a} sent{s}", label, f"{prefix}_article_{a}"))
    return rows


def skewed_label(label, seed=0):
    """Few long articles + a long tail of stubs, like a real wiki scrape."""
    rng = random.Random(seed)
    sizes = [rng.randint(80, 200) for _ in range(5)] + [rng.randint(1, 3) for _ in range(300)]
    return make_rows(label, sizes)


def share_of(train, test, label):
    n_test = sum(r[1] == label for r in test)
    n_all = n_test + sum(r[1] == label for r in train)
    return n_test / n_all


# ---------- assert_no_leak ----------

def test_no_leak_passes_on_disjoint_split():
    train = [("a", "x", "s1"), ("b", "y", "s2")]
    test = [("c", "x", "s3")]
    assert assert_no_leak(train, test) is True


def test_no_leak_catches_planted_overlap():
    train = [("shared sentence", "other", ""), ("b", "ligurian", "s2")]
    test = [("shared sentence", "other", ""), ("c", "ligurian", "s3")]
    with pytest.raises(AssertionError, match="other=1"):
        assert_no_leak(train, test, stage="test")


def test_no_leak_is_class_agnostic():
    # Same text under a different label still counts: it was trained on.
    train = [("identical text", "venetian", "s1")]
    test = [("identical text", "ligurian", "s2")]
    with pytest.raises(AssertionError):
        assert_no_leak(train, test)


def test_no_leak_reproduces_old_other_bug():
    # The original product_config bug: test-`other` was a subset of the train pool.
    pool = [(f"tatoeba {i}", "other", "") for i in range(50)]
    with pytest.raises(AssertionError, match="other=10"):
        assert_no_leak(pool, pool[:10])


# ---------- per_label_group_split ----------

def test_no_article_spans_both_sides():
    rows = skewed_label("ligurian") + skewed_label("piedmontese", seed=1)
    train, test = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    train_src = {r[2] for r in train}
    test_src = {r[2] for r in test}
    assert train_src.isdisjoint(test_src)


def test_every_row_kept_exactly_once():
    rows = skewed_label("ligurian")
    train, test = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    assert sorted(train + test) == sorted(rows)


@pytest.mark.parametrize("seed", range(10))
def test_row_ratio_near_target_on_skewed_articles(seed):
    rows = skewed_label("ligurian", seed=seed)
    train, test = per_label_group_split(rows, 0.2, seed=seed, verbose=False)
    assert abs(share_of(train, test, "ligurian") - 0.2) < 0.02


def test_groupshufflesplit_drifts_on_same_data():
    """Documents WHY the custom split exists: on the same skewed data,
    sklearn's GroupShuffleSplit (20% of groups) misses 20% of rows badly."""
    GroupShuffleSplit = pytest.importorskip("sklearn.model_selection").GroupShuffleSplit
    rows = skewed_label("ligurian")
    groups = [r[2] for r in rows]
    worst = 0.0
    for seed in range(20):
        _, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                     .split(rows, groups=groups))
        worst = max(worst, abs(len(te) / len(rows) - 0.2))
    assert worst > 0.10


def test_each_label_hits_target_independently():
    rows = (skewed_label("ligurian")
            + make_rows("friulian", [2] * 500)          # many short articles
            + make_rows("emilian", [1] * 300))          # all stubs
    train, test = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    for label in ("ligurian", "friulian", "emilian"):
        assert abs(share_of(train, test, label) - 0.2) < 0.02, label


def test_deterministic_for_same_input():
    rows = skewed_label("ligurian") + skewed_label("sicilian", seed=3)
    a = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    b = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    assert a == b


def test_independent_of_label_block_order():
    # Seeded per label, so the order labels appear in doesn't change any split.
    # (Row order WITHIN a label does matter: the committed CSVs fix it.)
    lig, sic = skewed_label("ligurian"), skewed_label("sicilian", seed=3)
    _, a_test = per_label_group_split(lig + sic, 0.2, seed=0, verbose=False)
    _, b_test = per_label_group_split(sic + lig, 0.2, seed=0, verbose=False)
    assert sorted(a_test) == sorted(b_test)


def test_single_group_label_goes_to_train():
    rows = make_rows("ladin", [40])   # one article only: nothing can be held out
    train, test = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    assert len(test) == 0 and len(train) == 40


def test_rows_without_source_are_not_pooled():
    # Sourceless rows (e.g. Tatoeba) must be split individually,
    # not treated as one giant pseudo-article.
    rows = [(f"s{i}", "other", "") for i in range(100)]
    train, test = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    assert len(test) == 20


def test_split_indices_matches_row_split():
    rows = skewed_label("ligurian")
    sents, labels, sources = zip(*rows)
    tr_idx, te_idx = split_indices(sents, labels, sources, 0.2, seed=0, verbose=False)
    _, test = per_label_group_split(rows, 0.2, seed=0, verbose=False)
    assert sorted(rows[i] for i in te_idx) == sorted(test)
    assert set(tr_idx).isdisjoint(te_idx)


# ---------- the committed split files ----------

def read_split(name):
    def load(path):
        with open(path, encoding="utf-8", newline="") as f:
            return [(r["sentence"], r["label"], r.get("source", "")) for r in csv.DictReader(f)]
    base = ROOT / "data" / "configs"
    return load(base / f"{name}_train.csv"), load(base / f"{name}_test.csv")


def test_committed_product_v2_split_is_leak_free():
    train, test = read_split("product_v2")
    assert assert_no_leak(train, test, stage="product_v2")


def test_committed_product_v2_other_languages_disjoint():
    # Product v2 holds out whole Tatoeba languages for `other`; no shared source key.
    train, test = read_split("product_v2")
    tr = {r[2] for r in train if r[1] == "other" and r[2]}
    te = {r[2] for r in test if r[1] == "other" and r[2]}
    assert tr.isdisjoint(te)


def test_leak_check_catches_legacy_product_config():
    # README claim: the hardened check catches the original bug on the old files.
    train, test = read_split("product_config")
    with pytest.raises(AssertionError, match="other="):
        assert_no_leak(train, test, stage="legacy product_config")
