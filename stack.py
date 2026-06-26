"""
Stacking combiner: can a meta-learner recover the complementary errors?
=======================================================================
The two models disagree on 72% of their errors (only one wrong at a time).
This script tests whether a small meta-classifier over their probability
vectors can convert that disagreement into correct answers.

HONESTY: a stacker trained and tested on the same eval rows would cheat. So we
estimate its true performance with STRATIFIED 5-FOLD CROSS-VALIDATION — every
prediction is made by a meta-model that never saw that sentence. The number it
prints is leakage-free. (To actually SHIP a stacker you'd train it on a held-out
set, not the eval; this script only answers "is there recoverable headroom?")

Compares five combiners on testset_eval_clean.csv:
  n-gram alone | XLM-R alone | flat average | confidence-weighted | CV-stacked

    python stack.py --ngram dialect_ngram.pt --xlmr dialect_xlmr --test testset_eval_clean.csv
"""

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--ngram", default="dialect_ngram.pt")
ap.add_argument("--xlmr", default="dialect_xlmr")
ap.add_argument("--test", default="testset_eval_clean.csv")
ap.add_argument("--max-len", type=int, default=128)
args = ap.parse_args()

CANON = ["lombard", "neapolitan", "sicilian", "standard", "venetian"]
name_to_id = {c: i for i, c in enumerate(CANON)}


def read_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    has_src = len(rows[0]) >= 3
    return [(r[0], r[1]) for r in rows[1:] if len(r) >= 2]


class DialectNet(nn.Module):
    def __init__(self, in_dim, hidden, n):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(hidden, n)
    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.fc1(x))))


def align(probs, model_classes):
    return probs[:, [model_classes.index(c) for c in CANON]]


# ---- data ----
data = read_csv(args.test)
texts = [d[0] for d in data]
y = np.array([name_to_id[d[1]] for d in data])

# ---- n-gram probs ----
b = torch.load(args.ngram, map_location="cpu", weights_only=False)
ng = DialectNet(b["input_dim"], b["hidden"], len(b["classes"])); ng.load_state_dict(b["state_dict"]); ng.eval()
Xn = torch.from_numpy(b["vectorizer"].transform(texts).toarray().astype(np.float32))
with torch.no_grad():
    p_ng = align(torch.softmax(ng(Xn), 1).numpy(), list(b["classes"]))

# ---- transformer probs ----
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(args.xlmr)
xl = AutoModelForSequenceClassification.from_pretrained(args.xlmr)
dev = "cuda" if torch.cuda.is_available() else "cpu"; xl.to(dev).eval()
xl_classes = [xl.config.id2label[i] for i in range(len(CANON))]
chunks = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i+64], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
        chunks.append(torch.softmax(xl(**enc).logits, 1).cpu().numpy())
p_xl = align(np.concatenate(chunks), xl_classes)


def macro(pred):
    return f1_score(y, pred, average="macro", zero_division=0)


# ---- baseline combiners (no training) ----
res = {}
res["n-gram alone"] = macro(p_ng.argmax(1))
res["XLM-R alone"] = macro(p_xl.argmax(1))
res["flat average"] = macro(((p_ng + p_xl) / 2).argmax(1))

# confidence-weighted: weight each model by its own max prob, per sentence
w_ng = p_ng.max(1, keepdims=True); w_xl = p_xl.max(1, keepdims=True)
conf = (w_ng * p_ng + w_xl * p_xl) / (w_ng + w_xl)
res["confidence-weighted"] = macro(conf.argmax(1))

# ---- CV-stacked (leakage-free) ----
X = np.hstack([p_ng, p_xl])                 # 10 features per sentence
oof = np.zeros(len(y), dtype=int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_scores = []
for tr, te in skf.split(X, y):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict(X[te])
    fold_scores.append(f1_score(y[te], oof[te], average="macro", zero_division=0))
res["CV-stacked"] = macro(oof)

# ---- report ----
print("\n" + "=" * 56)
print("CROSS-DOMAIN macro-F1 by combiner")
for k in ["n-gram alone", "XLM-R alone", "flat average", "confidence-weighted", "CV-stacked"]:
    print(f"  {k:<22}: {res[k]:.4f}")
print("=" * 56)
print(f"CV-stack per-fold: {[round(s,3) for s in fold_scores]}  (spread shows stability)")
best = max(res, key=res.get)
print(f"\nBest: {best} ({res[best]:.4f}) vs n-gram alone ({res['n-gram alone']:.4f}) "
      f"-> {res[best]-res['n-gram alone']:+.4f}")
if res[best] - res["n-gram alone"] < 0.005:
    print("Verdict: no meaningful lift. The complementary errors aren't separable "
          "from the probability vectors -> task wall confirmed, ship the n-gram model.")
else:
    print("Verdict: real lift. The stacker recovers some disagreement -> worth shipping "
          "(train it on a held-out split, not the eval).")
