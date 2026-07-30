"""
Shippable 3-model stacker  (leakage-free: meta trained on held-out val)
=======================================================================
Converts the cross-validated 0.97 ESTIMATE into a number you can claim.

Honest separation:
  * meta-training set = the in-domain VAL split (800 Wikipedia rows), reproduced
    with the SAME per-label row-aware split (split_utils.SPLIT_SEED) the base
    models used. Neither base
    model trained its weights on these rows.
  * test set = testset_eval_clean.csv, scored EXACTLY ONCE at the end.

Trains a LogisticRegression over the two base models' probability vectors, saves
the whole ensemble to --save, and reports n-gram / XLM-R / stacker on the test.

CAVEAT printed in the output: the meta-model is trained on FORMAL (Wikipedia)
register and tested on CASUAL register. If the stacker holds above the n-gram
on the casual test, it generalizes and is worth shipping. If it drops below,
stacking doesn't transfer across register -> ship the plain n-gram.

    python stack_train.py --data balanced_clean.csv --ngram dialect_ngram.pt \
        --xlmr dialect_xlmr --test testset_eval_clean.csv --save stacker.joblib
"""

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="balanced_clean.csv")
ap.add_argument("--ngram", default="dialect_ngram.pt")
ap.add_argument("--xlmr", default="dialect_xlmr")
ap.add_argument("--test", default="testset_eval_clean.csv")
ap.add_argument("--save", default="stacker.joblib")
ap.add_argument("--max-len", type=int, default=128)
args = ap.parse_args()

# CANON is derived from the trained n-gram bundle rather than hardcoded — it
# used to be a fixed 6-class list, which silently KeyError'd the moment the
# class set changed (expansion to 11/13 classes). Deriving it keeps this script
# config-agnostic: it works for the ITDI-parity 11-class and the product
# 13-class runs without edits, and guarantees the stacker's class order matches
# the base model it's actually stacking on.
def read3(path):
    return [(r[0], r[1], r[2] if len(r) >= 3 else "")
            for r in list(csv.reader(open(path, encoding="utf-8")))[1:] if len(r) >= 2]


class DialectNet(nn.Module):
    def __init__(self, d, h, n):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(h, n)
    def forward(self, x): return self.fc2(self.drop(self.relu(self.fc1(x))))


# ---- load base models once ----
b = torch.load(args.ngram, map_location="cpu", weights_only=False)
ngram_classes = list(b["classes"])
ng = DialectNet(b["input_dim"], b["hidden"], len(ngram_classes))
ng.load_state_dict(b["state_dict"]); ng.eval()
ng_vec = b["vectorizer"]

CANON = sorted(ngram_classes)
nid = {c: i for i, c in enumerate(CANON)}
print(f"Class set ({len(CANON)}) derived from {args.ngram}: {CANON}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(args.xlmr)
xl = AutoModelForSequenceClassification.from_pretrained(args.xlmr)
dev = "cuda" if torch.cuda.is_available() else "cpu"; xl.to(dev).eval()
xl_classes = [xl.config.id2label[i] for i in range(xl.config.num_labels)]

# both base models must cover the same classes, or align() would silently
# mis-map probability columns and the stacker would train on garbage
if set(xl_classes) != set(CANON):
    raise SystemExit(
        f"CLASS SET MISMATCH — the two base models were trained on different classes.\n"
        f"  n-gram ({len(CANON)}): {CANON}\n"
        f"  XLM-R  ({len(xl_classes)}): {sorted(xl_classes)}\n"
        f"Retrain whichever is stale on the same config before stacking.")


def align(p, classes): return p[:, [classes.index(c) for c in CANON]]


def base_probs(texts):
    Xn = torch.from_numpy(ng_vec.transform(texts).toarray().astype(np.float32))
    with torch.no_grad():
        p_ng = align(torch.softmax(ng(Xn), 1).numpy(), ngram_classes)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i+64], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(dev)
            chunks.append(torch.softmax(xl(**enc).logits, 1).cpu().numpy())
    p_xl = align(np.concatenate(chunks), xl_classes)
    return np.hstack([p_ng, p_xl]), p_ng, p_xl


# ---- META-TRAIN set: reproduce in-domain val split (held out from base weights) ----
data = read3(args.data)
skipped = {d[1] for d in data if d[1] not in nid}
if skipped:
    print(f"WARNING: --data contains labels outside the model's class set, skipping: {sorted(skipped)}")
    data = [d for d in data if d[1] in nid]
sent = np.array([d[0] for d in data], dtype=object)
src = np.array([d[2] for d in data], dtype=object)
lab = [d[1] for d in data]
y_all = np.array([nid[d[1]] for d in data])

# Reproduce the EXACT val split the base models held out. This is what makes the
# stacker leakage-free: the meta-model must train on rows neither base model fit
# its weights on. Both base scripts call this same helper with the same
# (test_size=0.2, seed=SPLIT_SEED) — if those ever diverge, the stacker silently
# meta-trains on base-model training rows and its score becomes meaningless.
from split_utils import split_indices, SPLIT_SEED
_, va = split_indices(sent, lab, src, test_size=0.2, seed=SPLIT_SEED, verbose=False)
va = np.array(va)
val_texts = list(sent[va]); val_y = y_all[va]
Xval, _, _ = base_probs(val_texts)
print(f"meta-train on {len(val_texts)} held-out val rows (formal register)")

meta = LogisticRegression(max_iter=2000, C=1.0)
meta.fit(Xval, val_y)

# ---- TEST once ----
test = read3(args.test)
skipped_t = {t[1] for t in test if t[1] not in nid}
if skipped_t:
    print(f"WARNING: --test contains labels outside the model's class set, skipping: {sorted(skipped_t)}")
    test = [t for t in test if t[1] in nid]
t_texts = [t[0] for t in test]; t_y = np.array([nid[t[1]] for t in test])
Xt, p_ng_t, p_xl_t = base_probs(t_texts)
pred_stack = meta.predict(Xt)


def macro(p): return f1_score(t_y, p, average="macro", zero_division=0)


print("\n" + "=" * 56)
print("CROSS-DOMAIN macro-F1 on testset (scored once)")
print(f"  n-gram alone : {macro(p_ng_t.argmax(1)):.4f}")
print(f"  XLM-R alone  : {macro(p_xl_t.argmax(1)):.4f}")
print(f"  STACKER      : {macro(pred_stack):.4f}   (trained on held-out val)")
print("=" * 56)
print("\n=== STACKER per-class ===")
print(classification_report(t_y, pred_stack, labels=list(range(len(CANON))),
                            target_names=CANON, zero_division=0))

ng_f = macro(p_ng_t.argmax(1)); st_f = macro(pred_stack)
if st_f >= ng_f + 0.003:
    print(f"VERDICT: stacker generalizes across register (+{st_f-ng_f:.4f}) -> SHIP IT.")
    joblib.dump({"meta": meta, "canon": CANON,
                 "ngram_path": args.ngram, "xlmr_path": args.xlmr}, args.save)
    print(f"Saved 3-model ensemble -> {args.save}")
else:
    print(f"VERDICT: stacker does NOT beat n-gram on casual test ({st_f:.4f} vs {ng_f:.4f}). "
          f"Register transfer failed -> ship the plain n-gram model. Not saving stacker.")