# ---------------------------------------------------------------------------
# LEGACY (v1, 6-class). Kept for project history — not part of the current
# pipeline; expects the old testset_eval_clean.csv and class set.
#
# Simple probability-averaging ensemble plus the complementarity diagnostic
# (do the two models fail on the SAME sentences or different ones?). That
# diagnostic is what motivated the learned stacker — averaging weights both
# models equally, while stack_train.py learns per-class weights and beat it.
# ---------------------------------------------------------------------------

"""
Ensemble: from-scratch n-gram net  +  XLM-R transformer
=======================================================
Inference only — no training. Loads both already-saved models, runs each over
testset_eval_clean.csv, averages class probabilities, and reports:
  * each model's cross-domain macro-F1
  * the ENSEMBLE macro-F1 (prob average)
  * the key diagnostic: do the two models fail on DIFFERENT sentences
    (complementary -> ensemble helps) or the SAME ones (shared task wall)?

Class columns are aligned BY NAME, so the two models' label orderings can never
silently mismatch.

PREREQUISITES (save both models first, one-time, on the 3060):
    python transformer.py --data balanced_clean.csv --extra-train testset_train_clean.csv \
        --test testset_eval_clean.csv --batch 8 --grad-accum 2 --save dialect_xlmr
    python model.py --data balanced_clean.csv --extra-train testset_train_clean.csv \
        --test testset_eval_clean.csv --features both --save dialect_ngram.pt

USAGE:
    python ensemble.py --ngram dialect_ngram.pt --xlmr dialect_xlmr \
        --test testset_eval_clean.csv
"""

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--ngram", default="dialect_ngram.pt")
ap.add_argument("--xlmr", default="dialect_xlmr")
ap.add_argument("--test", default="testset_eval_clean.csv")
ap.add_argument("--max-len", type=int, default=128)
args = ap.parse_args()


def read_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    has_src = len(rows[0]) >= 3
    return [(r[0], r[1], r[2] if has_src and len(r) >= 3 else "")
            for r in rows[1:] if len(r) >= 2]


# ---- from-scratch net (must match model.py's DialectNet exactly) ----
class DialectNet(nn.Module):
    def __init__(self, in_dim, hidden, n):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(hidden, n)
    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.fc1(x))))


def align(probs, model_classes, canon):
    """Reorder prob columns from model_classes order -> canonical order, by name."""
    idx = [model_classes.index(c) for c in canon]
    return probs[:, idx]


# ---- load data ----
data = read_csv(args.test)
texts = [d[0] for d in data]
gold_names = [d[1] for d in data]
canon = ["lombard", "neapolitan", "sicilian", "standard", "venetian"]
name_to_id = {c: i for i, c in enumerate(canon)}
y = np.array([name_to_id[g] for g in gold_names])

# ---- n-gram model probs ----
bundle = torch.load(args.ngram, map_location="cpu", weights_only=False)
ng_classes = list(bundle["classes"])
ng = DialectNet(bundle["input_dim"], bundle["hidden"], len(ng_classes))
ng.load_state_dict(bundle["state_dict"]); ng.eval()
vec = bundle["vectorizer"]
Xn = torch.from_numpy(vec.transform(texts).toarray().astype(np.float32))
with torch.no_grad():
    p_ngram = align(torch.softmax(ng(Xn), 1).numpy(), ng_classes, canon)

# ---- transformer probs ----
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(args.xlmr)
xlmr = AutoModelForSequenceClassification.from_pretrained(args.xlmr)
dev = "cuda" if torch.cuda.is_available() else "cpu"
xlmr.to(dev).eval()
xlmr_classes = [xlmr.config.id2label[i] for i in range(len(canon))]
p_list = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt").to(dev)
        p_list.append(torch.softmax(xlmr(**enc).logits, 1).cpu().numpy())
p_xlmr = align(np.concatenate(p_list), xlmr_classes, canon)

# ---- predictions ----
pred_ngram = p_ngram.argmax(1)
pred_xlmr = p_xlmr.argmax(1)
pred_ens = ((p_ngram + p_xlmr) / 2).argmax(1)


def macro(p):
    return f1_score(y, p, average="macro", zero_division=0)


# ---- headline ----
print("\n" + "=" * 56)
print("CROSS-DOMAIN macro-F1")
print(f"  n-gram (from scratch): {macro(pred_ngram):.3f}")
print(f"  XLM-R  (transformer) : {macro(pred_xlmr):.3f}")
print(f"  ENSEMBLE (prob avg)  : {macro(pred_ens):.3f}   (baseline to beat 0.96)")
print("=" * 56)

# ---- the diagnostic: complementary vs shared errors ----
ng_wrong = pred_ngram != y
xl_wrong = pred_xlmr != y
both_wrong = ng_wrong & xl_wrong
only_ng = ng_wrong & ~xl_wrong
only_xl = xl_wrong & ~ng_wrong
print("\nERROR OVERLAP (out of", len(y), "eval sentences):")
print(f"  both models wrong (shared wall) : {both_wrong.sum()}")
print(f"  only n-gram wrong (XLM-R saves) : {only_ng.sum()}")
print(f"  only XLM-R wrong (n-gram saves) : {only_xl.sum()}")
total_err = (ng_wrong | xl_wrong).sum()
shared = both_wrong.sum() / max(total_err, 1)
print(f"  -> {shared:.0%} of all errors are SHARED.")
if shared > 0.7:
    print("     High overlap = same task wall; ensemble won't help much. "
          "Strong 'it's the task, not the model' evidence for the write-up.")
else:
    print("     Low overlap = complementary errors; the ensemble has real headroom.")

print("\n=== ENSEMBLE per-class ===")
print(classification_report(y, pred_ens, target_names=canon, zero_division=0))
