"""
Prep the 'other' class for the 6-class retrain
===============================================
Samples other_data.csv into:
  * balanced_6class.csv  — balanced_clean.csv + 691 'other' sentences (matches dialect class sizes)
  * other_eval.csv       — 500 held-out 'other' sentences to verify rejection works

    python prep_other.py
"""

import csv, random, os
random.seed(0)

BALANCED   = "balanced_clean.csv"
OTHER_RAW  = "other_data.csv"
OUT_TRAIN  = "balanced_6class.csv"
OUT_EVAL   = "other_eval.csv"
TRAIN_N    = 691   # match dialect class sizes
EVAL_N     = 500

def read(p):
    return list(csv.reader(open(p, encoding="utf-8")))[1:]

def write(p, header, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# load
balanced = read(BALANCED)
other    = read(OTHER_RAW)
random.shuffle(other)

assert len(other) >= TRAIN_N + EVAL_N, \
    f"Need {TRAIN_N+EVAL_N} other sentences, only have {len(other)}"

train_other = other[:TRAIN_N]
eval_other  = other[TRAIN_N:TRAIN_N + EVAL_N]

# write balanced_6class.csv
header = ["sentence", "label", "source"]
write(OUT_TRAIN, header, balanced + train_other)

# write other_eval.csv
write(OUT_EVAL, header, eval_other)

# verify
import collections
rows_out = read(OUT_TRAIN)
counts = collections.Counter(r[1] for r in rows_out)
print(f"balanced_6class.csv: {len(rows_out)} rows")
print("  class counts:", dict(sorted(counts.items())))
print(f"other_eval.csv: {len(eval_other)} rows")
print("All OK — ready to retrain.")
