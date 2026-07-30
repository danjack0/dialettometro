"""
predict.py — the usable dialect detector
=========================================
Type a sentence, get a dialect + confidence. Works with either model:
  * the from-scratch n-gram net   (--model models/itdi_ngram_boost.pt)
  * the 3-model stacker (best)    (--model models/itdi_stacker.joblib
                                   --ngram models/itdi_ngram_boost.pt
                                   --xlmr models/itdi_xlmr)

CLASS SET IS DERIVED FROM THE MODEL, NOT HARDCODED. This used to be a fixed
5-name list, which silently broke the moment the class set changed (the
expansion to the 11-class ITDI-parity and 13-class product configs). Now the
labels come from the bundle itself, so one script serves every config and a
mismatched pair of base models fails loudly instead of mis-mapping probability
columns.

CONFIDENCE FLOOR: if the top probability is below --floor, the tool answers
"uncertain" instead of forcing a label. Even the 13-class product model only
knows a fraction of the ~30 varieties of Italy, so this stops it confidently
mislabelling an unseen dialect. Tune with --calibrate.

Note: the product config has a trained `other` rejection class, so for that
model the floor is a second line of defence rather than the only one. The
11-class ITDI-parity model has NO rejection class (that's what makes it
benchmark-comparable), so the floor matters much more there.

USAGE
-----
    # single sentence, n-gram model
    python predict.py --model models/itdi_ngram_boost.pt "La Sicilia è na bella ìsula"

    # best model (stacker)
    python predict.py --model models/product_stacker.joblib \
        --ngram models/product_ngram_boost.pt --xlmr models/product_xlmr "..."

    # interactive
    python predict.py --model models/product_ngram_boost.pt

    # tune the floor on a held-out set
    python predict.py --model models/product_ngram_boost.pt \

REMOTE MODELS (Hugging Face Hub)
---------------------------------
--xlmr already accepts a Hub repo ID directly (e.g. "danjack0/dialettometro-
product") — that's built into transformers' from_pretrained(), no change
needed here. --model and --ngram are custom .pt/.joblib files, which are NOT
natively Hub-aware, so pass --hf-repo to fetch them as named files from a Hub
repo instead of expecting a local path:

    python predict.py --hf-repo danjack0/dialettometro-product \
        --model product_stacker.joblib --ngram product_ngram_boost.pt \
        --xlmr danjack0/dialettometro-product "..."

Downloaded files are cached locally by huggingface_hub (~/.cache/huggingface)
so repeated runs don't re-download.
        --calibrate data/configs/product_config_test.csv
"""

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn


def resolve(path, hf_repo=None):
    """If hf_repo is given, treat `path` as a filename inside that Hub repo
    and download it (cached after the first time). Otherwise `path` is used
    exactly as before — a local file path. This is the only thing that
    changes between local and Hub-hosted use; everything downstream is
    identical either way."""
    if not hf_repo:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=hf_repo, filename=path)


class DialectNet(nn.Module):
    def __init__(self, d, h, n):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(h, n)
    def forward(self, x): return self.fc2(self.drop(self.relu(self.fc1(x))))


class Predictor:
    """Wraps either the plain n-gram net or the full stacker behind one .proba().

    Importable: itdi_eval.py and app.py both build on this rather than keeping
    their own copies of the model-loading logic.

    After construction, `self.canon` is the ordered class list and every
    probability matrix returned is in that order.
    """

    def __init__(self, model, ngram=None, xlmr=None, max_len=128, hf_repo=None):
        self.mode = "stacker" if str(model).endswith(".joblib") else "ngram"
        ng_path = ngram if self.mode == "stacker" else model
        if self.mode == "stacker" and not ng_path:
            raise SystemExit("stacker mode needs --ngram (the base n-gram bundle)")

        b = torch.load(resolve(ng_path, hf_repo), map_location="cpu", weights_only=False)
        self.ng_classes = list(b["classes"])
        self.ng = DialectNet(b["input_dim"], b["hidden"], len(self.ng_classes))
        self.ng.load_state_dict(b["state_dict"]); self.ng.eval()
        self.ng_vec = b["vectorizer"]
        self.max_len = max_len

        if self.mode == "ngram":
            # single model: its own class order is the canonical order
            self.canon = list(self.ng_classes)
            return

        import joblib
        bundle = joblib.load(resolve(model, hf_repo))
        self.meta = bundle["meta"]
        # the stacker stores the exact class order its meta-model was fit on;
        # trust that, but verify the supplied n-gram bundle matches it
        self.canon = list(bundle.get("canon") or sorted(self.ng_classes))
        if set(self.canon) != set(self.ng_classes):
            raise SystemExit(
                f"CLASS SET MISMATCH between stacker and n-gram bundle.\n"
                f"  stacker ({len(self.canon)}): {self.canon}\n"
                f"  n-gram  ({len(self.ng_classes)}): {sorted(self.ng_classes)}\n"
                f"You have probably paired an ITDI-parity model with a product "
                f"model, or a stale bundle. Pass the matching --ngram.")

        if not xlmr:
            raise SystemExit("stacker mode needs --xlmr (the base transformer dir)")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(xlmr)
        self.xl = AutoModelForSequenceClassification.from_pretrained(xlmr)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.xl.to(self.dev).eval()
        # num_labels, NOT len(canon) — if the transformer was fine-tuned on a
        # different class set this must surface as an error, not a silent slice
        self.xl_classes = [self.xl.config.id2label[i]
                           for i in range(self.xl.config.num_labels)]
        if set(self.xl_classes) != set(self.canon):
            raise SystemExit(
                f"CLASS SET MISMATCH between stacker and XLM-R model.\n"
                f"  stacker ({len(self.canon)}): {self.canon}\n"
                f"  XLM-R   ({len(self.xl_classes)}): {sorted(self.xl_classes)}\n"
                f"Pass the matching --xlmr directory.")

    def _align(self, p, classes):
        """Reorder a probability matrix from `classes` order into canon order."""
        return p[:, [classes.index(c) for c in self.canon]]

    def _ng_probs(self, texts):
        X = torch.from_numpy(self.ng_vec.transform(texts).toarray().astype(np.float32))
        with torch.no_grad():
            return self._align(torch.softmax(self.ng(X), 1).numpy(), self.ng_classes)

    def _xl_probs(self, texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = self.tok(texts[i:i + 64], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.dev)
                out.append(torch.softmax(self.xl(**enc).logits, 1).cpu().numpy())
        return self._align(np.concatenate(out), self.xl_classes)

    def proba(self, texts):
        """Return an (N, len(canon)) probability matrix in canon order."""
        p_ng = self._ng_probs(texts)
        if self.mode == "ngram":
            return p_ng
        feats = np.hstack([p_ng, self._xl_probs(texts)])
        proba = self.meta.predict_proba(feats)
        out = np.zeros((len(texts), len(self.canon)))
        for j, c in enumerate(list(self.meta.classes_)):
            out[:, int(c)] = proba[:, j]
        return out

    def ranked(self, probs_row):
        return sorted(zip(self.canon, probs_row.tolist()), key=lambda x: -x[1])


def answer(pred_obj, row, floor):
    ranking = pred_obj.ranked(row)
    top_label, top_p = ranking[0]
    if top_p < floor:
        return (f"uncertain — not a recognized dialect "
                f"(top guess {top_label} {top_p:.0%}, below floor {floor:.0%})")
    others = ", ".join(f"{l} {p:.0%}" for l, p in ranking[1:3])
    return f"{top_label}  ({top_p:.0%})   [next: {others}]"


def calibrate(pred_obj, path):
    """Show confidence of correct vs wrong predictions to pick a floor."""
    rows = [(r[0], r[1]) for r in list(csv.reader(open(path, encoding="utf-8")))[1:]
            if len(r) >= 2]
    known = set(pred_obj.canon)
    skipped = {r[1] for r in rows if r[1] not in known}
    if skipped:
        print(f"NOTE: skipping rows with labels this model doesn't know: {sorted(skipped)}")
        rows = [r for r in rows if r[1] in known]
    if not rows:
        raise SystemExit("no rows left to calibrate on")

    texts = [r[0] for r in rows]; gold = [r[1] for r in rows]
    P = pred_obj.proba(texts)
    conf = P.max(1); pred = P.argmax(1)
    correct = np.array([pred_obj.canon[pred[i]] == gold[i] for i in range(len(rows))])
    cc, wc = conf[correct], conf[~correct]
    print(f"\nCalibration on {len(rows)} rows (acc {correct.mean():.3f}):")
    print(f"  CORRECT predictions: confidence mean {cc.mean():.3f}, "
          f"5th pct {np.percentile(cc, 5):.3f}")
    if len(wc):
        print(f"  WRONG   predictions: confidence mean {wc.mean():.3f}, "
              f"95th pct {np.percentile(wc, 95):.3f}")
    print("\n  floor | kept | of-kept-correct | rejected-that-were-wrong")
    for f in [0.4, 0.5, 0.6, 0.7, 0.8]:
        keep = conf >= f
        kept_acc = correct[keep].mean() if keep.any() else float("nan")
        caught = (~correct & ~keep).sum() / max((~correct).sum(), 1)
        print(f"   {f:.2f} | {keep.mean():.0%} | {kept_acc:.3f}          | {caught:.0%}")
    print("\nPick the floor that keeps most real sentences while catching most errors.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="an n-gram .pt or a stacker .joblib")
    p.add_argument("--ngram", default=None, help="base n-gram bundle (stacker mode)")
    p.add_argument("--xlmr", default=None, help="base transformer dir (stacker mode) "
                   "— accepts a local path OR a Hugging Face Hub repo ID directly")
    p.add_argument("--hf-repo", default=None,
                   help="Hugging Face Hub repo ID to fetch --model/--ngram from "
                        "(e.g. danjack0/dialettometro-product). Omit to use local paths.")
    p.add_argument("--floor", type=float, default=0.55,
                   help="confidence floor below which the answer is 'uncertain'")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--calibrate", default=None, help="CSV to tune the floor on, then exit")
    p.add_argument("text", nargs="*", help="sentence to classify (omit for interactive)")
    args = p.parse_args()

    pred = Predictor(args.model, args.ngram, args.xlmr, args.max_len, args.hf_repo)
    print(f"loaded {pred.mode} model | {len(pred.canon)} classes: {pred.canon}")

    if args.calibrate:
        calibrate(pred, args.calibrate); return

    if args.text:
        sentence = " ".join(args.text)
        print(answer(pred, pred.proba([sentence])[0], args.floor))
        return

    print("Type a sentence (blank line to quit).")
    while True:
        try:
            s = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s:
            break
        print("   " + answer(pred, pred.proba([s])[0], args.floor))


if __name__ == "__main__":
    main()