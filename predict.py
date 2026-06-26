"""
predict.py — the usable dialect detector
=========================================
Type a sentence, get a dialect + confidence. Works with either model:
  * the from-scratch n-gram net      (--model dialect_ngram.pt)
  * the 3-model stacker (best, 0.974) (--model stacker.joblib --ngram dialect_ngram.pt --xlmr dialect_xlmr)

CONFIDENCE FLOOR: if the top probability is below --floor, the tool answers
"uncertain / not a recognized dialect" instead of forcing one of the 5 labels.
Your model only knows 5 of ~30+ Italian dialects, so this stops it confidently
mislabelling English, other dialects, or junk. Tune --floor with --calibrate.

USAGE
-----
    # single sentence
    python predict.py --model dialect_ngram.pt "La Sicilia è na bella ìsula"

    # best model (stacker)
    python predict.py --model stacker.joblib --ngram dialect_ngram.pt --xlmr dialect_xlmr "..."

    # interactive
    python predict.py --model dialect_ngram.pt

    # find a good floor from your eval (prints confidence distribution of right vs wrong)
    python predict.py --model dialect_ngram.pt --calibrate testset_eval_clean.csv
"""

import argparse
import csv
import sys
import numpy as np
import torch
import torch.nn as nn

CANON = ["lombard", "neapolitan", "sicilian", "standard", "venetian"]


class DialectNet(nn.Module):
    def __init__(self, d, h, n):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(h, n)
    def forward(self, x): return self.fc2(self.drop(self.relu(self.fc1(x))))


def align(p, classes):
    return p[:, [classes.index(c) for c in CANON]]


class Predictor:
    """Wraps either the plain n-gram net or the full stacker behind one .proba()."""
    def __init__(self, args):
        self.mode = "stacker" if args.model.endswith(".joblib") else "ngram"
        # n-gram base (needed in both modes)
        ng_path = args.ngram if self.mode == "stacker" else args.model
        b = torch.load(ng_path, map_location="cpu", weights_only=False)
        self.ng_classes = list(b["classes"])
        self.ng = DialectNet(b["input_dim"], b["hidden"], len(self.ng_classes))
        self.ng.load_state_dict(b["state_dict"]); self.ng.eval()
        self.ng_vec = b["vectorizer"]

        if self.mode == "stacker":
            import joblib
            bundle = joblib.load(args.model)
            self.meta = bundle["meta"]
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tok = AutoTokenizer.from_pretrained(args.xlmr)
            self.xl = AutoModelForSequenceClassification.from_pretrained(args.xlmr)
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.xl.to(self.dev).eval()
            self.xl_classes = [self.xl.config.id2label[i] for i in range(len(CANON))]
        self.max_len = args.max_len

    def _ng_probs(self, texts):
        X = torch.from_numpy(self.ng_vec.transform(texts).toarray().astype(np.float32))
        with torch.no_grad():
            return align(torch.softmax(self.ng(X), 1).numpy(), self.ng_classes)

    def _xl_probs(self, texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = self.tok(texts[i:i+64], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.dev)
                out.append(torch.softmax(self.xl(**enc).logits, 1).cpu().numpy())
        return align(np.concatenate(out), self.xl_classes)

    def proba(self, texts):
        """Return (N, 5) class-probability matrix in CANON order."""
        p_ng = self._ng_probs(texts)
        if self.mode == "ngram":
            return p_ng
        p_xl = self._xl_probs(texts)
        # meta is a LogisticRegression -> predict_proba over [p_ng || p_xl]
        feats = np.hstack([p_ng, p_xl])
        proba = self.meta.predict_proba(feats)
        # map meta classes (ints) back to CANON columns
        cols = list(self.meta.classes_)
        out = np.zeros((len(texts), len(CANON)))
        for j, c in enumerate(cols):
            out[:, c] = proba[:, j]
        return out


def ranked(probs_row):
    return sorted(zip(CANON, probs_row.tolist()), key=lambda x: -x[1])


def answer(pred, floor):
    top_label, top_p = ranked(pred)[0]
    if top_p < floor:
        return f"uncertain — not a recognized dialect (top guess {top_label} {top_p:.0%}, below floor {floor:.0%})"
    others = ", ".join(f"{l} {p:.0%}" for l, p in ranked(pred)[1:3])
    return f"{top_label}  ({top_p:.0%})   [next: {others}]"


def calibrate(pred_obj, path):
    """Show confidence of correct vs wrong predictions to pick a floor."""
    rows = [(r[0], r[1]) for r in list(csv.reader(open(path, encoding="utf-8")))[1:] if len(r) >= 2]
    texts = [r[0] for r in rows]
    gold = [r[1] for r in rows]
    P = pred_obj.proba(texts)
    conf = P.max(1); pred = P.argmax(1)
    correct = np.array([CANON[pred[i]] == gold[i] for i in range(len(rows))])
    cc, wc = conf[correct], conf[~correct]
    print(f"\nCalibration on {len(rows)} rows (acc {correct.mean():.3f}):")
    print(f"  CORRECT predictions: confidence mean {cc.mean():.3f}, 5th pct {np.percentile(cc,5):.3f}")
    print(f"  WRONG   predictions: confidence mean {wc.mean():.3f}, 95th pct {np.percentile(wc,95):.3f}")
    print("\n  floor | kept | of-kept-correct | rejected-that-were-wrong")
    for f in [0.4, 0.5, 0.6, 0.7, 0.8]:
        keep = conf >= f
        kept_acc = correct[keep].mean() if keep.any() else float("nan")
        caught = (~correct & ~keep).sum() / max((~correct).sum(), 1)
        print(f"   {f:.2f} | {keep.mean():.0%} | {kept_acc:.3f}          | {caught:.0%}")
    print("\nPick the floor that keeps most real sentences while catching most errors. "
          "0.5–0.6 is usually the sweet spot here.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="dialect_ngram.pt or stacker.joblib")
    p.add_argument("--ngram", default="dialect_ngram.pt", help="base n-gram (stacker mode)")
    p.add_argument("--xlmr", default="dialect_xlmr", help="base transformer dir (stacker mode)")
    p.add_argument("--floor", type=float, default=0.55, help="confidence floor for 'uncertain'")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--calibrate", default=None, help="CSV to tune the floor on, then exit")
    p.add_argument("text", nargs="*", help="sentence to classify (omit for interactive)")
    args = p.parse_args()

    pred = Predictor(args)

    if args.calibrate:
        calibrate(pred, args.calibrate); return

    if args.text:
        sentence = " ".join(args.text)
        print(answer(pred.proba([sentence])[0], args.floor))
        return

    print("Type a sentence (blank line to quit).")
    while True:
        try:
            s = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s:
            break
        print("   " + answer(pred.proba([s])[0], args.floor))


if __name__ == "__main__":
    main()
