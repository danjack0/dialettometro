"""
audit_split_skew.py
===================
Read-only audit: how far does each class's test share drift from the 20% target
under sklearn's GroupShuffleSplit vs. this project's per_label_group_split?

Both split by source article (no article spans train and test). The difference
is what they target: GroupShuffleSplit takes 20% of GROUPS; per_label_group_split
fills 20% of each label's ROWS.

Run from the repo root:
    python scripts/audit_split_skew.py            # 50 seeds
    python scripts/audit_split_skew.py --seeds 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import per_label_group_split  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "cleaned" / "balanced_13class.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    d = pd.read_csv(DATA)
    labels = sorted(d.label.unique())
    totals = d.label.value_counts()
    rows = list(d[["sentence", "label", "source"]].itertuples(index=False, name=None))

    gss = {l: [] for l in labels}
    ours = {l: [] for l in labels}
    for seed in range(args.seeds):
        _, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                       random_state=seed).split(d, groups=d.source))
        t = d.iloc[te].label.value_counts()
        _, te_rows = per_label_group_split(rows, args.test_size, seed, verbose=False)
        o = pd.Series([r[1] for r in te_rows]).value_counts()
        for l in labels:
            gss[l].append(t.get(l, 0) / totals[l])
            ours[l].append(o.get(l, 0) / totals[l])

    print(f"Test share per class over {args.seeds} seeds (target {args.test_size:.0%})\n")
    print(f"{'label':13}{'rows':>6}{'articles':>9} | {'GroupShuffleSplit':>17} | {'per_label_group_split':>21}")
    for l in labels:
        g, o = np.array(gss[l]), np.array(ours[l])
        n_art = d[d.label == l].source.nunique()
        print(f"{l:13}{totals[l]:6}{n_art:9} | {g.min():7.1%} – {g.max():6.1%} | {o.min():10.1%} – {o.max():6.1%}")
    worst_g = max(abs(x - args.test_size) for l in labels for x in gss[l])
    worst_o = max(abs(x - args.test_size) for l in labels for x in ours[l])
    print(f"\nWorst deviation from target: GroupShuffleSplit {worst_g:.1%}, per_label_group_split {worst_o:.1%}")


if __name__ == "__main__":
    main()
