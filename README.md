# Dialettometro

An Italian regional dialect classifier covering 11 varieties, built as an
independent replication of the [VarDial 2022 ITDI shared task](https://github.com/noe-eva/ITDI_2022),
with a product-oriented extension for standalone use.

**Try it:** `python scripts/app.py --model models/product_ngram_boost.pt` → `http://localhost:5000`

---

## What this is

Two models trained from one shared, audited dataset:

| | **ITDI-parity** | **Product** |
|---|---|---|
| Classes | 11 (matches the shared task exactly) | 13 (11 dialects + Standard Italian + "other" rejection) |
| Purpose | Direct comparison against VarDial 2022 | Usable standalone tool |
| Standard/other | Excluded | Included |

**Dialects covered:** Emilian-Romagnol, Friulian, Ladin, Ligurian, Lombard,
Neapolitan, Piedmontese, Sardinian, Sicilian, Tarantino, Venetian.

Both models are 3-way stacked ensembles: a from-scratch character+word n-gram
network, a fine-tuned XLM-RoBERTa transformer, and a logistic-regression
meta-model that combines them — trained and validated with strict
article-level splitting so no source document ever leaks between train and
test.

## Results

### Held-out test (own Wikipedia-derived split, article-level, never seen in training)

| | ITDI-parity (11-class) | Product (13-class) |
|---|---|---|
| n-gram baseline | 0.8331 | 0.8411 |
| XLM-R | 0.8175 | 0.8507 |
| **Stacker** | **0.8699** | **0.8767** |

All scores are macro-F1. The stacker beats both base models in both configs —
the two architectures fail on different sentences (n-gram wins on
fine-grained dialect-vs-dialect distinctions; XLM-R wins on
dialect-vs-standard-vs-other), so stacking recovers cases neither model gets
alone. Ligurian in particular jumps from ~0.63 F1 (either base model alone) to
~0.78 (stacker).

### External benchmark: VarDial 2022 ITDI held-out set (eval-only, never trained on)

The ITDI 2022 organizers' dev/test files were used strictly to score the
trained model — never for training, never redistributed (see
[Data & Licensing](#data--licensing)).

| | dev.txt (7 of 11 classes present) | test_gold_standard.txt (8 of 11 classes present) |
|---|---|---|
| Weighted-F1 (shared-task headline) | **0.7865** | **0.5707** |
| Macro-F1 (present classes) | 0.7928 | 0.5564 |
| Official baseline — fastText | 0.1322 | 0.1322 |
| Official baseline — SVM unigram | 0.4899 | 0.4899 |
| Official baseline — SVM char n-gram (best) | 0.7726 | 0.7726 |

Dev and test each cover a different partial subset of the 11 classes, so
neither is directly comparable to the model's own 11-class held-out score
above — different label set, different (casual, "sources unknown") register.
On dev, the model ties the organizers' strongest baseline. On test, it lands
between the SVM-unigram and SVM-char-n-gram baselines.

### What the external benchmark revealed that the internal split couldn't

Two real, explainable weaknesses surfaced only by testing against genuinely
external, casual-register text:

- **Ligurian flips failure mode under domain shift.** On the model's own
  held-out Wikipedia test: precision 0.80 / recall 0.51 (under-fires). On
  ITDI's casual text: precision 0.46-0.53 / recall 0.75-0.95 (over-fires,
  absorbing Venetian and Sicilian sentences). The class-weight boost that
  helped the Wikipedia score appears to make the model too eager to fire
  Ligurian on ambiguous casual text specifically.
- **Tarantino nearly collapses on real casual text**: F1 0.94-0.96 on the
  model's own held-out set vs. **0.21** on ITDI. True Tarantino sentences
  scatter to Neapolitan and Sicilian almost as often as they're caught
  correctly — consistent with Tarantino's real dialectological position as a
  transitional variety between Neapolitan and Sicilian-Salentino. The
  Wikipedia-trained signal is encyclopedic-register-specific and doesn't
  transfer to spoken/casual text.

### Known limitation: Ladin

Precision stays high (0.97-1.00) but recall collapses (0.24-0.42 depending on
config) — the model learned a narrow, confident signature from the ~1,250
training rows available (Ladin's Wikipedia is small) and won't fire on
anything outside it. This is a data-volume ceiling, not a bug; boosting its
loss weight helps but has a hard limit.

## Project structure

```
data/
  raw/       scraped Wikipedia + Tatoeba text, pre-cleaning
  titles/    harvested article title lists, one per source wiki
  cleaned/   post-audit, balanced dataset (balanced_13class.csv)
  configs/   final train/test splits for both configs
scripts/     full pipeline: harvest → scrape → audit → balance → split →
             train (n-gram, transformer, stacker) → predict → serve
models/      trained model artifacts (n-gram bundles, XLM-R dirs, stackers)
index.html   frontend, served by scripts/app.py
```

## Pipeline

1. `harvest_titles.py` — pulls real article titles per wiki via the
   MediaWiki API, excluding redirects and near-empty stubs
2. `scraper.py` / `tatoeba.py` — fetch article text / Tatoeba sentences
3. `audit_junk.py` — two-pass content audit: no-signal junk (ISBNs,
   citations, foreign-language contamination) and template/near-duplicate
   bot-boilerplate detection, both blind to any model's predictions
4. `prepare_expansion_dataset.py` — applies the audit, reduces template
   boilerplate, caps oversized classes with source-diversity-aware sampling
5. `split_configs.py` / `split_utils.py` — builds both class configs with a
   row-count-aware, per-label, article-level train/test split (standard
   `GroupShuffleSplit` samples by group *count*, which badly skews splits
   when article lengths vary — this project uses a custom split that targets
   the row ratio instead)
6. `model.py` — from-scratch char+word n-gram network, with optional
   balanced class weighting and per-class loss boosting
7. `transformer.py` — fine-tunes XLM-RoBERTa with the same split and
   weighting conventions
8. `stack_train.py` — logistic-regression meta-model over both base models'
   probabilities, meta-trained on the exact validation rows the base models
   held out (leakage-free by construction)
9. `predict.py` — CLI + importable inference, with a confidence floor for
   rejecting unrecognized input
10. `app.py` / `index.html` — Flask API + frontend; both derive their class
    list from the loaded model, not a hardcoded list, so the UI never falls
    out of sync when the class set changes
11. `itdi_eval.py` — eval-only scoring against the official VarDial 2022
    ITDI benchmark

## Setup

```bash
pip install -r requirements.txt
```

Train both configs end-to-end (see `scripts/` docstrings for full flag
reference):

```bash
python scripts/model.py --data data/configs/product_config_train.csv \
    --test data/configs/product_config_test.csv --features both \
    --class-weights balanced --boost ladin=2.5 --boost ligurian=2.0 \
    --save models/product_ngram_boost.pt

python scripts/transformer.py --data data/configs/product_config_train.csv \
    --test data/configs/product_config_test.csv --class-weights balanced \
    --boost ladin=2.5 --boost ligurian=2.0 --save models/product_xlmr

python scripts/stack_train.py --data data/configs/product_config_train.csv \
    --ngram models/product_ngram_boost.pt --xlmr models/product_xlmr \
    --test data/configs/product_config_test.csv --save models/product_stacker.joblib
```

Run the app:

```bash
python scripts/app.py --model models/product_stacker.joblib \
    --ngram models/product_ngram_boost.pt --xlmr models/product_xlmr --floor 0.60
```

(The n-gram model alone — `--model models/product_ngram_boost.pt`, no
`--ngram`/`--xlmr` — is a lightweight, fast alternative that doesn't need the
~1.1GB transformer loaded.)

## Data & Licensing

See [SOURCES.md](SOURCES.md) for full data provenance. In short: training
data comes from Wikipedia dumps (CC-BY-SA) and Tatoeba (CC-BY 2.0 FR),
fetched fresh via each script rather than redistributed. The VarDial 2022
ITDI dev/test files (no license file in their repo, provenance described only
as "sources unknown by participants") were used **strictly as a held-out
external benchmark** — scored against, never trained on, never
redistributed. Prediction files derived from them are excluded from this repo
via `.gitignore`.