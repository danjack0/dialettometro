"""
Italian Dialect Classifier — TRANSFORMER fine-tuning phase
==========================================================
Fine-tunes a pretrained multilingual transformer (XLM-RoBERTa by default,
mBERT optional) on the SAME 5-way task and the SAME splits as model.py, so the
result is directly comparable to the from-scratch baseline:

    cross-domain macro-F1 = 0.96   (the number to beat)
    in-domain   macro-F1 = 0.90

Mirrors model.py exactly:
  * source-aware split: split_utils.per_label_group_split(test_size=0.2) on the
    `source` column of --data  -> in-domain train / in-domain val (unseen articles)
  * --extra-train mixed WHOLESALE into training (teaches casual register)
  * --test held out, scored once as the cross-domain scoreboard
  * same confusion-matrix + per-class P/R/F1 report format
  * headline metric = CROSS-DOMAIN MACRO-F1 (eval set is ~42% Sicilian; raw
    accuracy is misleading)

Difference from baseline (intentional, per project plan): best-model selection
and early stopping use in-domain val MACRO-F1 (not val loss).

USAGE
-----
    # XLM-R (recommended first run)
    python transformer.py --data balanced.csv --extra-train testset_train.csv \
        --test testset_eval.csv

    # mBERT comparison (cheaper)
    python transformer.py --data balanced.csv --extra-train testset_train.csv \
        --test testset_eval.csv --model bert-base-multilingual-cased

    # save the fine-tuned model for inference
    python transformer.py ... --save dialect_xlmr/

DEPENDENCIES
------------
    pip install "transformers>=4.41" torch scikit-learn numpy
    (XLM-R tokenizer also needs:  pip install sentencepiece )

The first run downloads the base model from Hugging Face (~1.1 GB for XLM-R),
so the machine running this needs internet access for that download.
"""

import argparse
import csv
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="balanced.csv")
ap.add_argument("--extra-train", default=None)
ap.add_argument("--test", default=None)
ap.add_argument("--model", default="xlm-roberta-base",
                help="HF model id. Try bert-base-multilingual-cased for mBERT.")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--boost", action="append", default=[], metavar="CLASS=FACTOR",
                help="multiply one class's loss weight, e.g. --boost ladin=2.0. Repeatable. "
                     "Mirrors model.py. Only keep it if MACRO-F1 improves.")
ap.add_argument("--class-weights", default="none", choices=["none", "balanced"],
                help="balanced = inverse-frequency loss weights, computed on the FINAL "
                     "training set (after --extra-train mixing). Mirrors model.py so the "
                     "two base models stay methodologically comparable.")
ap.add_argument("--batch", type=int, default=16,
                help="per-device train batch. Lower to 8 on a 6 GB laptop GPU.")
ap.add_argument("--grad-accum", type=int, default=1,
                help="gradient accumulation steps (raise if you lower --batch).")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-len", type=int, default=128,
                help="token cap; sentences here are short, 128 is plenty.")
ap.add_argument("--patience", type=int, default=2,
                help="early-stopping patience on val macro-F1 (epochs).")
ap.add_argument("--output", default="hf_out",
                help="working dir for checkpoints/logs.")
ap.add_argument("--errors", default=None,
                help="CSV path to dump every misclassified sentence (val + test) "
                     "with true/pred label, source, and model confidence.")
ap.add_argument("--save", default=None,
                help="dir to save the final fine-tuned model + tokenizer.")
args = ap.parse_args()


def read_csv(path):
    """Same loader as model.py: tolerant of a missing source column."""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    has_src = len(rows[0]) >= 3
    return [(r[0], r[1], r[2] if has_src and len(r) >= 3 else "")
            for r in rows[1:] if len(r) >= 2]


# ---------------------------------------------------------------- 1. LOAD
data = read_csv(args.data)
sentences = np.array([d[0] for d in data], dtype=object)
labels = [d[1] for d in data]
sources = np.array([d[2] for d in data], dtype=object)
classes = sorted(set(labels))                      # same order as model.py
class_to_id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in class_to_id.items()}
y = np.array([class_to_id[l] for l in labels])
print(f"Loaded {len(sentences)} sentences | classes: {classes} | model: {args.model}")

# ---------------------------------------------------------------- 2. SPLIT
# source-aware AND row-count-aware, per label (identical call to model.py).
# sklearn's GroupShuffleSplit samples a fraction of GROUPS, not rows, which with
# uneven article sizes produced badly skewed val sets and corrupted early
# stopping. See split_utils.py.
if any(s for s in sources) and len(set(sources)) > len(classes):
    from split_utils import split_indices, SPLIT_SEED
    tr_idx, va_idx = split_indices(sentences, labels, sources,
                                   test_size=0.2, seed=SPLIT_SEED)
    tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
    print(f">> Honest split: validating on UNSEEN articles "
          f"({len(set(sources[va_idx]))} held out).")
else:
    from sklearn.model_selection import train_test_split as _tts
    tr_idx, va_idx = _tts(np.arange(len(y)), test_size=0.2,
                          stratify=y, random_state=0)

train_txt = list(sentences[tr_idx]); train_y = list(y[tr_idx])
val_txt = list(sentences[va_idx]);   val_y = list(y[va_idx])

# 2b. MIX casual training data wholesale (teaches register)
if args.extra_train:
    extra = read_csv(args.extra_train)
    ex_txt = [e[0] for e in extra if e[1] in class_to_id]
    ex_y = [class_to_id[e[1]] for e in extra if e[1] in class_to_id]
    train_txt += ex_txt; train_y += ex_y
    print(f">> Mixed {len(ex_txt)} casual sentences into TRAINING.")

print(f"   train={len(train_txt)}  in-domain val={len(val_txt)}")

# ---------------------------------------------------------------- 3. TOKENIZE
tokenizer = AutoTokenizer.from_pretrained(args.model)


class DS(torch.utils.data.Dataset):
    """Tokenizes lazily; dynamic padding is handled by the collator."""
    def __init__(self, texts, ys):
        self.enc = tokenizer(list(texts), truncation=True, max_length=args.max_len)
        self.y = list(ys)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item


train_ds = DS(train_txt, train_y)
val_ds = DS(val_txt, val_y)
collator = DataCollatorWithPadding(tokenizer)

# ---------------------------------------------------------------- 4. MODEL
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(classes),
    id2label=id2label, label2id=class_to_id)


def compute_metrics(eval_pred):
    logits, labels_ = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "macro_f1": f1_score(labels_, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels_, preds),
    }


# ---------------------------------------------------------------- 5. TRAIN
use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()   # Ampere (30xx) supports bf16
use_fp16 = use_cuda and not use_bf16
print(f"Device: {'cuda' if use_cuda else 'cpu'} | "
      f"{'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

ta_kwargs = dict(
    output_dir=args.output,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    warmup_ratio=0.1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    bf16=use_bf16,
    fp16=use_fp16,
    seed=SEED,
    report_to="none",
)
# transformers renamed evaluation_strategy -> eval_strategy in 4.41; support both.
try:
    targs = TrainingArguments(eval_strategy="epoch", save_strategy="epoch",
                              save_total_limit=1, **ta_kwargs)
except TypeError:
    targs = TrainingArguments(evaluation_strategy="epoch", save_strategy="epoch",
                              save_total_limit=1, **ta_kwargs)

# ---------------------------------------------------------------- 5b. CLASS WEIGHTS
# HF Trainer's default loss is unweighted. With the expanded class set spanning
# ~7x in size (friulian ~3200 vs emilian ~440), the thin classes get swamped.
# WeightedTrainer applies the same inverse-frequency weights model.py uses, so
# the two base models stay methodologically comparable.
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels_ = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = (self.class_weights.to(logits.device)
             if self.class_weights is not None else None)
        loss = torch.nn.functional.cross_entropy(logits, labels_, weight=w)
        return (loss, outputs) if return_outputs else loss


class_weights_t = None
if args.class_weights == "balanced":
    counts = np.bincount(np.array(train_y), minlength=len(classes)).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(train_y) / (len(classes) * counts)
    class_weights_t = torch.tensor(w, dtype=torch.float32)
    print(">> balanced class weights (inverse frequency):")
    for c in classes:
        i = class_to_id[c]
        print(f"     {c:>12}  n={int(counts[i]):>6}  weight={class_weights_t[i]:.3f}")

if args.boost:
    if class_weights_t is None:
        class_weights_t = torch.ones(len(classes), dtype=torch.float32)
    for spec in args.boost:
        if "=" not in spec:
            raise SystemExit(f"--boost expects CLASS=FACTOR, got '{spec}'")
        cname, factor = spec.rsplit("=", 1)
        if cname not in class_to_id:
            raise SystemExit(f"--boost: unknown class '{cname}'. Known: {classes}")
        class_weights_t[class_to_id[cname]] *= float(factor)
        print(f">> {cname} class loss weight boosted x{factor} "
              f"-> {class_weights_t[class_to_id[cname]]:.3f}")

TrainerCls = WeightedTrainer if class_weights_t is not None else Trainer
extra_kw = {"class_weights": class_weights_t} if class_weights_t is not None else {}

trainer = TrainerCls(
    model=model, args=targs,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collator, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    **extra_kw,
)
trainer.train()


# ---------------------------------------------------------------- 6. REPORT
def report(yt, yp, title):
    """Identical layout to model.py for side-by-side comparison."""
    print(f"\n=== {title} ===")
    print("         " + " ".join(f"{c[:4]:>5}" for c in classes))
    cm = confusion_matrix(yt, yp, labels=list(range(len(classes))))
    for i, c in enumerate(classes):
        print(f"{c[:8]:>8} " + " ".join(f"{cm[i][j]:5d}" for j in range(len(classes))))
    print("\n" + classification_report(yt, yp, labels=list(range(len(classes))),
                                        target_names=classes, zero_division=0))
    return f1_score(yt, yp, average="macro", zero_division=0)


# in-domain val (unseen Wikipedia articles)
val_logits = trainer.predict(val_ds).predictions
val_preds = np.argmax(val_logits, axis=1)
val_macro = report(val_y, val_preds,
                   "IN-DOMAIN validation (Wikipedia, unseen articles)")

# cross-domain held-out scoreboard
test_macro = None
t_logits = t_pred = t_y = t_txt = t_src = None
if args.test:
    test = read_csv(args.test)
    kept = [t for t in test if t[1] in class_to_id]
    t_txt = [t[0] for t in kept]
    t_y = [class_to_id[t[1]] for t in kept]
    t_src = [t[2] for t in kept]
    if t_txt:
        test_ds = DS(t_txt, t_y)
        t_logits = trainer.predict(test_ds).predictions
        t_pred = np.argmax(t_logits, axis=1)
        test_acc = accuracy_score(t_y, t_pred)
        print(f"\nCROSS-DOMAIN accuracy: {test_acc:.3f} "
              f"(vs {accuracy_score(val_y, val_preds):.3f} in-domain)")
        test_macro = report(t_y, t_pred,
                            f"CROSS-DOMAIN test: {args.test} ({len(t_txt)} sentences)")

# ---------------------------------------------------------------- 7. SUMMARY
print("\n" + "=" * 56)
print("HEADLINE (compare to from-scratch baseline)")
print(f"  in-domain   macro-F1: {val_macro:.3f}   (baseline 0.90)")
if test_macro is not None:
    print(f"  CROSS-DOMAIN macro-F1: {test_macro:.3f}   (baseline 0.96  <- the number to beat)")
print("=" * 56)

if args.save:
    trainer.save_model(args.save)
    tokenizer.save_pretrained(args.save)
    print(f"\nSaved fine-tuned model + tokenizer -> {args.save}")


# ---------------------------------------------------------------- 8. ERROR DUMP
def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _collect(split, texts, ys, srcs, logits):
    """Return misclassified rows with confidences. Sorted most-confident-wrong first."""
    probs = _softmax(np.asarray(logits))
    preds = probs.argmax(axis=1)
    rows = []
    for i in range(len(ys)):
        if preds[i] == ys[i]:
            continue
        rows.append({
            "split": split,
            "source": srcs[i],
            "true": id2label[ys[i]],
            "pred": id2label[int(preds[i])],
            "p_true": round(float(probs[i, ys[i]]), 3),
            "p_pred": round(float(probs[i, preds[i]]), 3),
            "sentence": texts[i],
        })
    # most confident mistakes first -> these are the likely mislabels / hard cases
    rows.sort(key=lambda r: -r["p_pred"])
    return rows


if args.errors:
    all_rows = []
    all_rows += _collect("val", val_txt, val_y, list(sources[va_idx]), val_logits)
    if t_logits is not None:
        all_rows += _collect("test", t_txt, t_y, t_src, t_logits)

    with open(args.errors, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "split", "source", "true", "pred", "p_true", "p_pred", "sentence"])
        w.writeheader()
        w.writerows(all_rows)

    # concentration summary -> the tell for real headroom vs genuine overlap
    from collections import Counter
    cells = Counter((r["true"], r["pred"]) for r in all_rows if r["split"] == "test")
    srcs_c = Counter(r["source"] for r in all_rows if r["split"] == "test")
    n_test_err = sum(1 for r in all_rows if r["split"] == "test")
    print(f"\nWrote {len(all_rows)} misclassified sentences -> {args.errors}")
    print(f"  cross-domain errors: {n_test_err}")
    print("  top confusion cells (true -> pred):")
    for (t, p), n in cells.most_common(6):
        print(f"    {t:>10} -> {p:<10} {n}")
    print("  errors concentrated in sources:")
    for s, n in srcs_c.most_common(6):
        print(f"    {n:>3}  {s}")
    print("  (one source dominating = likely artifact/headroom; "
          "spread out = genuine overlap)")