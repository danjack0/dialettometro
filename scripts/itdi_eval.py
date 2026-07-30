"""
itdi_eval.py — score against the official VarDial 2022 ITDI held-out set
=======================================================================
EVAL ONLY. This script never trains on ITDI data and never copies it into the
project's own corpus. It reads their gold file, runs inference, writes a
prediction file, and prints metrics. Nothing else.

Why eval-only: the ITDI_2022 repo ships NO LICENSE file, and its dev/test are
"newly collected text samples (sources unknown by participants)" — unknown
provenance plus no licence grant means that data must not end up in a
redistributed corpus or baked into published model weights. Scoring against it
is the use the organisers published it for; training on it is not. Practical
rules that follow, and which this script is built around:
  * download the repo yourself, keep it OUTSIDE the project tree, .gitignore it
  * never commit their gold files or any {their sentence -> your prediction} dump
  * report aggregate metrics only, and cite the shared task

Get the data (do this yourself, outside the repo):
    git clone https://github.com/noe-eva/ITDI_2022.git

Usage:
    # n-gram model
    python itdi_eval.py --model models/itdi_ngram_boost.pt \
        --gold ../ITDI_2022/task/test_gold_standard.txt

    # stacker (best)
    python itdi_eval.py --model models/itdi_stacker.joblib \
        --ngram models/itdi_ngram_boost.pt --xlmr models/itdi_xlmr \
        --gold ../ITDI_2022/task/test_gold_standard.txt

Then, for the fully official number, run THEIR script on the emitted file:
    python ../ITDI_2022/task/ITDI_eval.py \
        ../ITDI_2022/task/test_gold_standard.txt itdi_predictions.txt

WHAT THE NUMBERS MEAN — read before quoting any of them
-------------------------------------------------------
1. Their dev and test cover DIFFERENT PARTIAL SUBSETS of the 11 varieties
   (dev: 7 classes, test: 8). So an ITDI score is NOT comparable to the
   project's own 11-class held-out macro-F1 — different label set, different
   register, different task shape. Quote them side by side, never as if one
   supersedes the other.
2. Their ITDI_eval.py scores over sorted(set(y_true + y_pred)). If the model
   predicts a class that has zero support in the gold file, that class enters
   the average as an automatic 0.0 and drags macro-F1 down. That is a real
   property of the official metric, not a bug — the official SVM baseline lost
   ~0.1 macro-F1 to exactly this. Both conventions are printed below so the
   effect is visible instead of silently absorbed.
3. Weighted-F1 is the headline the shared task reported; macro-F1 is this
   project's headline. Both are printed.

Official baselines on test_gold_standard.txt, for context:
    fastText          macro-F1 0.1004   weighted-F1 0.1322
    SVM unigram       macro-F1 0.3424   weighted-F1 0.4899   acc 0.4660
    SVM char n-gram   macro-F1 0.5193   weighted-F1 0.7726   acc 0.7522
"""

import argparse
import os
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

from predict import Predictor

# ITDI label code  ->  this project's class name
ITDI_TO_LOCAL = {
    "EML": "emilian",
    "FUR": "friulian",
    "LIJ": "ligurian",
    "LLD": "ladin",
    "LMO": "lombard",
    "NAP": "neapolitan",
    "PMS": "piedmontese",
    "ROA_TARA": "tarantino",
    "SC": "sardinian",
    "SCN": "sicilian",
    "VEC": "venetian",
}
LOCAL_TO_ITDI = {v: k for k, v in ITDI_TO_LOCAL.items()}


def read_gold(path):
    """ITDI format: one line per example, 'LABEL<TAB>sentence'."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                print(f"  skipping line {ln}: no tab separator")
                continue
            label, text = line.split("\t", 1)
            if not text.strip():
                continue
            rows.append((label.strip(), text))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="n-gram .pt or stacker .joblib")
    ap.add_argument("--ngram", default=None, help="base n-gram bundle (stacker mode)")
    ap.add_argument("--xlmr", default=None, help="base transformer dir (stacker mode)")
    ap.add_argument("--gold", required=True,
                    help="ITDI dev.txt or test_gold_standard.txt")
    ap.add_argument("--pred-out", default="itdi_predictions.txt",
                    help="prediction file in ITDI format, for their ITDI_eval.py")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--allow-non-itdi", action="store_true",
                    help="let the model predict classes outside the ITDI 11 "
                         "(e.g. the product model's `other`/`standard`). Off by "
                         "default: those labels do not exist in ITDI's space, so "
                         "predicting them is an automatic error and makes the "
                         "comparison meaningless rather than merely harder.")
    args = ap.parse_args()

    if not os.path.exists(args.gold):
        raise SystemExit(
            f"Gold file not found: {args.gold}\n"
            f"Clone it yourself, outside this repo:\n"
            f"  git clone https://github.com/noe-eva/ITDI_2022.git")

    pred = Predictor(args.model, args.ngram, args.xlmr)
    print(f"loaded {pred.mode} model | {len(pred.canon)} classes")

    rows = read_gold(args.gold)
    gold_codes = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f"gold: {len(rows)} rows from {args.gold}")

    unknown_codes = sorted({c for c in gold_codes if c not in ITDI_TO_LOCAL})
    if unknown_codes:
        raise SystemExit(f"gold file has unmapped label codes: {unknown_codes}")

    present = sorted(set(gold_codes))
    print(f"classes present in this gold file ({len(present)}): {present}")
    missing = sorted(set(ITDI_TO_LOCAL) - set(present))
    if missing:
        print(f"classes ABSENT from this gold file ({len(missing)}): {missing}")
        print("  -> this file scores a SUBSET of the 11-way task. Not comparable "
              "to the project's own 11-class held-out number.")

    # Which of the model's classes are valid ITDI labels
    itdi_cols = [i for i, c in enumerate(pred.canon) if c in LOCAL_TO_ITDI]
    non_itdi = [c for c in pred.canon if c not in LOCAL_TO_ITDI]
    if non_itdi:
        if args.allow_non_itdi:
            print(f"model has non-ITDI classes {non_itdi} — ALLOWED to fire "
                  f"(they will count as errors against every gold row).")
        else:
            print(f"model has non-ITDI classes {non_itdi} — masked out before "
                  f"argmax (--allow-non-itdi to disable).")
    model_missing = sorted(set(ITDI_TO_LOCAL.values()) - set(pred.canon))
    if model_missing:
        print(f"  WARNING: model cannot predict {model_missing} — those gold "
              f"rows can never be scored correctly.")

    # ---- inference ----
    preds_local = []
    for i in range(0, len(texts), args.batch):
        P = pred.proba(texts[i:i + args.batch])
        if not args.allow_non_itdi and non_itdi:
            mask = np.full(P.shape[1], -np.inf)
            mask[itdi_cols] = 0.0
            P = P + mask                     # -inf on non-ITDI columns
        preds_local.extend(pred.canon[j] for j in P.argmax(1))
        print(f"  scored {min(i + args.batch, len(texts))}/{len(texts)}", end="\r")
    print()

    pred_codes = [LOCAL_TO_ITDI.get(p, p.upper()) for p in preds_local]

    # ---- prediction file in ITDI's own format ----
    # Their ITDI_eval.py parses with pandas read_csv(sep="\t"), which hard-fails
    # if any line yields more than 2 fields. A handful of their own dev.txt rows
    # contain an embedded tab, so text is sanitised here to guarantee exactly
    # two fields per line. Their script only ever reads column 0 (the label), so
    # this cannot change the score.
    n_sanitised = sum(1 for t in texts if "\t" in t)
    with open(args.pred_out, "w", encoding="utf-8") as f:
        for code, text in zip(pred_codes, texts):
            f.write(f"{code}\t{text.replace(chr(9), ' ')}\n")
    print(f"wrote predictions -> {args.pred_out}")
    if n_sanitised:
        print(f"  ({n_sanitised} row(s) had embedded tabs replaced with spaces "
              f"so the file stays 2-column)")
    print(f"  NOTE: {args.pred_out} contains ITDI source text in column 2 — their "
          f"scorer needs a 2-column file. Keep it local; .gitignore it rather than "
          f"committing it, since the ITDI repo ships no licence.")

    # ---- metrics ----
    acc = accuracy_score(gold_codes, pred_codes)

    # (a) official convention: labels = sorted(set(gold + pred)), so a predicted
    #     class with zero gold support enters the macro average as 0.0
    official_labels = sorted(set(gold_codes) | set(pred_codes))
    macro_off = f1_score(gold_codes, pred_codes, labels=official_labels,
                         average="macro", zero_division=0)
    weighted_off = f1_score(gold_codes, pred_codes, labels=official_labels,
                            average="weighted", zero_division=0)

    # (b) present-only: macro over classes that actually appear in the gold
    macro_present = f1_score(gold_codes, pred_codes, labels=present,
                             average="macro", zero_division=0)

    ghosts = sorted(set(pred_codes) - set(gold_codes))

    print("\n" + "=" * 62)
    print(f"ITDI EVAL — {os.path.basename(args.gold)}  ({len(rows)} rows, "
          f"{len(present)} gold classes)")
    print("=" * 62)
    print(f"  accuracy                      : {acc:.4f}")
    print(f"  macro-F1  (official labelset) : {macro_off:.4f}")
    print(f"  weighted-F1 (official)        : {weighted_off:.4f}   <- shared-task headline")
    print(f"  macro-F1  (gold classes only) : {macro_present:.4f}   <- more meaningful")
    if ghosts:
        print(f"\n  predicted {len(ghosts)} class(es) with zero gold support: {ghosts}")
        print(f"  that costs {macro_present - macro_off:.4f} macro-F1 under the "
              f"official convention.")
    print("\n  official baselines (test_gold_standard.txt):")
    print("    fastText         macro 0.1004  weighted 0.1322")
    print("    SVM unigram      macro 0.3424  weighted 0.4899  acc 0.4660")
    print("    SVM char n-gram  macro 0.5193  weighted 0.7726  acc 0.7522")
    print("=" * 62)

    print("\n=== per-class (gold classes only) ===")
    print(classification_report(gold_codes, pred_codes, labels=present,
                                target_names=present, zero_division=0, digits=4))

    print("=== confusion matrix (rows = gold) ===")
    all_pred_labels = present + ghosts
    cm = confusion_matrix(gold_codes, pred_codes, labels=all_pred_labels)
    print("            " + " ".join(f"{c[:6]:>7}" for c in all_pred_labels))
    for i, c in enumerate(present):
        print(f"{c:>10}  " + " ".join(f"{cm[i][j]:7d}" for j in range(len(all_pred_labels))))

    print("\nFor the fully official number, run the organisers' own script:")
    print(f"  python ITDI_eval.py {args.gold} {args.pred_out}")
    if n_sanitised:
        print("  NOTE: this gold file contains embedded tabs, so their pandas-based "
              "ITDI_eval.py will fail to parse the GOLD side and error out before "
              "scoring anything. That is a pre-existing bug in their script, not in "
              "the predictions. test_gold_standard.txt is clean and works fine; for "
              "this file, the metrics printed above are the reliable path — they "
              "replicate the official label-set convention exactly.")
    print("\nReminder: do not commit the ITDI gold files or this prediction file "
          "(it embeds their sentences). Add both to .gitignore.")


if __name__ == "__main__":
    main()