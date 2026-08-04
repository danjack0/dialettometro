"""
split_utils.py
==============
ONE implementation of the source-aware split, imported by split_configs.py,
model.py, transformer.py and stack_train.py.

Why this exists
---------------
sklearn's GroupShuffleSplit picks a fraction of GROUPS, not rows. When a label's
articles vary wildly in length (normal for wiki scrapes: a few long articles
plus a long tail of stubs), taking 20% of groups can yield anywhere from 5% to
75% of that label's rows. Observed in this project: a global GroupShuffleSplit
gave ligurian a 76% test share in one run and 4.7% in another, and inside
model.py it produced a validation set where ligurian had 1753 rows vs emilian's
55 — so early stopping was selecting on a broken signal and the product config
restored "best" at epoch 1 (i.e. barely trained).

What this does instead
----------------------
Per label, independently: bucket that label's rows by source article, shuffle
the groups, then best-fit fill the held-out side by ROW count until the target
fraction is met. Whole groups always move together, so no article ever spans
both sides (no leakage), but each class lands close to the intended ratio.

Determinism: seeded per label via Random(f"{seed}:{label}") so results are
reproducible and independent of label iteration order.

CRITICAL for stack_train.py: it must reproduce model.py's exact val split to
meta-train on rows the base models never fit. Both import this same function
with the same (test_size, seed), which is what guarantees that.
"""

import random

# The canonical in-script validation seed. model.py, transformer.py and
# stack_train.py MUST all use this same value: stack_train.py meta-trains on the
# validation rows, which is only leakage-free if it reproduces the exact split
# the base models held out. Changing this invalidates any saved stacker.
SPLIT_SEED = 0


def group_key(row, i):
    """Source article for a row; rows with no source get a unique singleton key
    so they're never silently pooled together into one giant pseudo-article."""
    src = row[2] if len(row) >= 3 else ""
    return src if src else f"__no_source_{i}__"


def per_label_group_split(rows_in, test_size=0.2, seed=0, verbose=True):
    """Split rows into (train_rows, test_rows), targeting test_size by ROW count
    within each label, keeping whole source groups intact.

    rows_in: list of (sentence, label, source) tuples/lists.
    Returns: (train_rows, test_rows) preserving the input row objects.
    """
    by_label = {}
    for i, r in enumerate(rows_in):
        by_label.setdefault(r[1], []).append((i, r))

    train_rows, test_rows = [], []
    for label, items in sorted(by_label.items()):
        groups = {}
        for i, r in items:
            groups.setdefault(group_key(r, i), []).append(r)

        if len(groups) < 2:
            train_rows.extend(r for rs in groups.values() for r in rs)
            if verbose:
                print(f"    WARNING: label '{label}' has only {len(groups)} distinct "
                      f"source group(s) — all rows placed in train, none held out.")
            continue

        total = sum(len(rs) for rs in groups.values())
        target = total * test_size

        keys = list(groups)
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(keys)

        # best-fit: add whole groups that don't overshoot, largest first
        test_keys, acc = set(), 0
        for k in sorted(keys, key=lambda k: -len(groups[k])):
            if acc + len(groups[k]) <= target:
                test_keys.add(k)
                acc += len(groups[k])

        # still well short? take the one remaining group landing closest to target
        if acc < target * 0.6:
            remaining = [k for k in keys if k not in test_keys]
            if remaining:
                best = min(remaining, key=lambda k: abs((acc + len(groups[k])) - target))
                if abs((acc + len(groups[best])) - target) < abs(acc - target):
                    test_keys.add(best)
                    acc += len(groups[best])

        for k, rs in groups.items():
            (test_rows if k in test_keys else train_rows).extend(rs)

        if verbose:
            achieved = acc / total if total else 0
            if abs(achieved - test_size) > 0.07:
                print(f"    NOTE: label '{label}' held-out fraction {achieved:.1%} vs target "
                      f"{test_size:.0%} — rows concentrated in few large articles, "
                      f"no leak-free split can hit the target exactly.")
            if len(test_keys) <= 3:
                print(f"    NOTE: label '{label}' held-out split drawn from only "
                      f"{len(test_keys)} source article(s) — high-variance score.")
    return train_rows, test_rows


def assert_no_leak(train_rows, test_rows, stage="final"):
    """Fail loudly if ANY test sentence also appears in train, in ANY class.

    This is the check that the old split_configs.leak_check missed: it only
    inspected the article-level dialect split and ran BEFORE the `other` rows
    were appended, so a test sentence that was also in the train pool (the whole
    `other` bug: every test-`other` row was a copy of a train-`other` row) sailed
    straight through. Run this on the FINAL train/test — after every append — so
    nothing can be added behind the checker's back.

    Sentence-level and class-agnostic on purpose: a held-out row is leaked if its
    text was trained on at all, regardless of which label it carries.

    rows are (sentence, label, source, ...) — only slots 0 (sentence) and 1
    (label) are read. Raises AssertionError listing the offending classes.
    """
    train_sent = {r[0] for r in train_rows}
    leaked = {}
    for r in test_rows:
        if r[0] in train_sent:
            leaked[r[1]] = leaked.get(r[1], 0) + 1
    if leaked:
        total = sum(leaked.values())
        detail = ", ".join(f"{lab}={n}" for lab, n in sorted(leaked.items(), key=lambda x: -x[1]))
        raise AssertionError(
            f"LEAK [{stage}]: {total} test sentence(s) also appear in train "
            f"(by class: {detail}). A held-out row that was trained on invalidates "
            f"that class's score. Partition every class disjointly before writing "
            f"the split.")
    return True


def split_indices(sentences, labels, sources, test_size=0.2, seed=0, verbose=True):
    """Index-based wrapper for scripts that work with numpy arrays.

    Returns (train_idx, test_idx) as plain lists of ints into the input arrays.
    """
    rows = [(str(sentences[i]), labels[i], str(sources[i]) if sources is not None else "", i)
            for i in range(len(sentences))]
    # carry the original index in slot 3; group_key only reads slots 0-2
    train_rows, test_rows = per_label_group_split(rows, test_size, seed, verbose)
    return [r[3] for r in train_rows], [r[3] for r in test_rows]
