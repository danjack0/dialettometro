# Dialettometro

An Italian regional dialect classifier covering 11 varieties, built as an
independent replication of the [VarDial 2022 ITDI shared task](https://github.com/noe-eva/ITDI_2022),
with a product-oriented extension for standalone use.

**🔗 Live demo: [dialettometro.onrender.com](https://dialettometro.onrender.com)**
(free tier — the first request after idle takes ~30–60s to wake the instance).
The demo serves the 13-class product config; per `requirements-deploy.txt` it
runs the lightweight n-gram model (no transformer loaded).

**Run it yourself (no training needed):** trained models aren't committed to
git (they're large), but the product config is published **publicly** on the
Hugging Face Hub, so a fresh clone runs out of the box:

```bash
pip install -r requirements.txt
python scripts/app.py --hf-repo danjack0/dialettometro-product \
    --model product_ngram_boost.pt          # → http://localhost:5000
```

See [Setup](#setup) for the stacker (best model) and for training from scratch.

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

All scores are macro-F1. ("n-gram baseline" is the class-weight-boosted n-gram
— the same base model fed to the stacker — not the unboosted variant.) The
stacker beats both base models in both configs — the two architectures fail on
different sentences (n-gram wins on fine-grained dialect-vs-dialect
distinctions; XLM-R wins on dialect-vs-standard-vs-other), so stacking recovers
cases neither model gets alone. Ligurian in particular jumps from ~0.63 F1
(either base model alone) to ~0.78 (stacker).

**Caveat on the Product `other` class.** The product config's `other`
(rejection) test rows are **not** held out: all 500 test-`other` sentences also
appear in training, because the split appends the whole Tatoeba `other` pool to
train and a curated subset of that same pool to test. So the `other` class's own
apparent held-out performance is not a valid generalization estimate, and it
inflates the product macro-F1 by ≈0.006 (0.8767 with `other` vs 0.8708 over the
12 non-`other` classes). This affects **only** the `other` class — every
Wikipedia dialect class (and Standard Italian) is split strictly at the article
level with zero train/test leakage, in both configs.

### External benchmark: VarDial 2022 ITDI held-out set (eval-only, never trained on)

The ITDI 2022 organizers' dev/test files were used strictly to score the
trained model — never for training, never redistributed (see
[Data & Licensing](#data--licensing)).

| | dev.txt (7 of 11 classes present) | test_gold_standard.txt (8 of 11 classes present) |
|---|---|---|
| Weighted-F1 (shared-task headline) | **0.7865** | **0.5707** |
| Macro-F1 (present classes) | 0.7928 | 0.5564 |
| Official baseline — fastText | — | 0.1322 |
| Official baseline — SVM unigram | — | 0.4899 |
| Official baseline — SVM char n-gram (best) | — | 0.7726 |

The three official baseline scores are **test-set numbers only** (VarDial 2022,
Aepli et al., Table 3; support 11,087) — the organizers published no dev-set
baseline, so the dev column is left blank rather than repeating the test figures.

Dev and test each cover a different partial subset of the 11 classes, so
neither is directly comparable to the model's own 11-class held-out score
above — different label set, different (casual, "sources unknown") register.
On test, the model's weighted-F1 (0.5707) lands between the SVM-unigram
(0.4899) and SVM-char-n-gram (0.7726) baselines. The model's dev weighted-F1
(0.7865) is in the same range as that strongest baseline, but note the baseline
is a test-set score — there is no official dev baseline to compare against
directly.

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
  raw/       scraped + intermediate sentence CSVs, pre-final-cleaning
  titles/    harvested article title lists (12 source wikis; 15 files — a few
             wikis have a second harvest batch)
  cleaned/   post-audit, balanced dataset (balanced_13class.csv — 11 dialects
             + Standard Italian = 12 labels; the 13th class, `other`, is a
             Tatoeba-sourced rejection set appended when the product split is built)
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

`models/` is gitignored (the artifacts are large and reproducible), so a fresh
clone has no local model files. Pick one of the two options below.

### Option A — run a pretrained model from the Hub (no training)

The **product** config is published **publicly** at
[`danjack0/dialettometro-product`](https://huggingface.co/danjack0/dialettometro-product)
— one repo bundling the XLM-R weights, the n-gram bundle, and the stacker.
`--hf-repo` fetches the `.pt`/`.joblib` files by name; `--xlmr` takes the same
repo ID directly.

```bash
# lightweight: n-gram only (no ~1.1GB transformer download)
python scripts/app.py --hf-repo danjack0/dialettometro-product \
    --model product_ngram_boost.pt --floor 0.60

# best: full 3-model stacker
python scripts/app.py --hf-repo danjack0/dialettometro-product \
    --model product_stacker.joblib --ngram product_ngram_boost.pt \
    --xlmr danjack0/dialettometro-product --floor 0.60
```

The 11-class **ITDI-parity** config is **not** published to the Hub — train it
yourself with Option B.

### Option B — train from scratch

Both configs train end-to-end from the committed splits (see `scripts/`
docstrings for full flag reference):

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

Then run the app against the models you just trained:

```bash
python scripts/app.py --model models/product_stacker.joblib \
    --ngram models/product_ngram_boost.pt --xlmr models/product_xlmr --floor 0.60
```

(The n-gram model alone — `--model models/product_ngram_boost.pt`, no
`--ngram`/`--xlmr` — is a lightweight, fast alternative that doesn't need the
~1.1GB transformer loaded.)

## Related work

- Aepli et al., *Findings of the VarDial Evaluation Campaign 2022* —
  [paper](https://aclanthology.org/2022.vardial-1.1/) · [data](https://github.com/noe-eva/ITDI_2022)

---

## Data & Licensing

See [SOURCES.md](SOURCES.md) for full data provenance.

**Code license (MIT) vs. data license.** The `LICENSE` file (MIT) covers the
**code only**. The datasets committed under `data/` are *not* MIT — they are
derivative works of their upstream sources and remain under those licenses:

- **Wikipedia-derived** dialect + Standard-Italian text — **CC-BY-SA 4.0**
  (attribution + share-alike). Source: the MediaWiki API of each edition.
- **Tatoeba-derived** casual-register and `other`-class sentences —
  **CC-BY 2.0 FR** (attribution). Source: the Tatoeba API.

**What is redistributed vs. fetched fresh.** This repo **does** commit derived,
**sentence-level** CSVs — the cleaned corpus (`data/cleaned/balanced_13class.csv`),
the train/test splits (`data/configs/`), and the intermediate/scraped sentence
files (`data/raw/`, including `combined_raw.csv`). It does **not** commit the
raw Wikipedia XML **dumps**; the harvest/scrape scripts re-fetch those from
source. Because the committed text is CC-BY-SA / CC-BY, redistributing it here
is permitted **with attribution and share-alike**, which this section provides.

**VarDial 2022 ITDI benchmark — eval-only.** The ITDI dev/test files ship with
no license file, and their provenance is described only as "sources unknown by
participants." They were used **strictly as a held-out external benchmark** —
scored against, never trained on. The organizers' repo is kept **outside** this
project tree, and prediction files derived from it (`itdi_*_preds.txt`) are
excluded via `.gitignore`.

The **active** pipeline data — `balanced_13class.csv`, the config splits,
`combined_raw.csv`, and every file feeding the shipped models — contains
**zero** ITDI dev/test sentences (verified by exact-sentence comparison).

Four unused legacy files from the project's early 5-6-class phase, which
contained 80 sentences overlapping the ITDI dev set, were removed from the
repo. That overlap traced to a shared upstream source (Sicilian Wikisource
public-domain literary texts), not to ITDI - see
[SOURCES.md](SOURCES.md#note-on-removed-legacy-files).