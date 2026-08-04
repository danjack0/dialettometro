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
| Split files | `data/configs/itdi_parity_*` | `data/configs/product_v2_*` (leak-free; supersedes `product_config_*`) |

**Dialects covered:** Emilian-Romagnol, Friulian, Ladin, Ligurian, Lombard,
Neapolitan, Piedmontese, Sardinian, Sicilian, Tarantino, Venetian.

Both models are 3-way stacked ensembles: a from-scratch character+word n-gram
network, a fine-tuned XLM-RoBERTa transformer, and a logistic-regression
meta-model that combines them — trained and validated with strict
article-level splitting so no source document leaks between train and test.
The `other` rejection class is additionally split **by language**, so no
Tatoeba language appears on both sides (see
[Fixed: the Product `other` class](#fixed-the-product-other-class-is-now-genuinely-held-out-product_v2)).
One known exception: the ITDI-parity split carries a single coincidental
duplicate Sicilian sentence (1 row of 6,391); `product_v2` has zero overlap in
any class.

## Results

### Held-out test (own Wikipedia-derived split, article-level, never seen in training)

| | ITDI-parity (11-class) | Product v2 (13-class, leak-free) |
|---|---|---|
| n-gram baseline | 0.8331 | 0.7554 |
| XLM-R | 0.8175 | 0.8188 |
| **Stacker** | **0.8699** | **0.8811** |

All scores are macro-F1. ("n-gram baseline" is the class-weight-boosted n-gram
— the same base model fed to the stacker — not the unboosted variant.) The
stacker beats both base models in both configs — the two architectures fail on
different sentences (n-gram wins on fine-grained dialect-vs-dialect
distinctions; XLM-R wins on rejecting non-target languages), so stacking
recovers cases neither model gets alone. On ITDI-parity, Ligurian jumps from
~0.63 F1 (either base model alone) to 0.78 (stacker); on product v2 it goes from
0.37–0.48 alone to 0.70 stacked.

The **product** numbers above are for the leak-free **`product_v2`** split. The
original `product_config` reported a higher n-gram figure (0.8411) only because
its `other` test set was not truly held out — see the next section.

### Fixed: the Product `other` class is now genuinely held out (`product_v2`)

The original `product_config` had a real leak (was issue D in
[VERIFIED_FACTS.md](VERIFIED_FACTS.md)): **all 500 test-`other` rows also
appeared in training**, because the split appended the whole Tatoeba `other`
pool to train and a curated subset of that same pool to test. The `other`
class's held-out score was therefore measured on sentences the model had
trained on.

**Fix** (`scripts/split_product_v2.py` → `data/configs/product_v2_*`): the
`other` pool is re-partitioned **disjointly, grouped by Tatoeba language**, so no
language — and no sentence — straddles train and test. Five languages
(Catalan, Croatian, Portuguese, Romanian, Spanish; 2,300 rows) are held out of
training entirely, making the test a genuine *unseen-language* rejection check.
A strengthened `split_utils.assert_no_leak` now runs on the final rows (after
every append) and fails loudly on any train/test sentence overlap in any class;
it catches the old bug on the existing files. Every non-`other` row is carried
over from `product_config` unchanged, so the only variable is the `other` split.

What the honest split reveals (all verified against the retrained
`models/product_v2_*`):

| model | `other` F1 | product macro-F1 | vs. old (leaky) |
|---|---|---|---|
| n-gram alone | 0.56 | **0.7554** | 0.8411 → **−0.086** |
| XLM-R alone | 0.96 | **0.8188** | 0.8507 → −0.032 |
| **stacker** | **0.96** | **0.8811** | 0.8767 → **+0.004** |

- The **n-gram alone collapses**: it can't reject languages it never saw
  (unseen Romance text scatters into the dialect classes — Ligurian precision
  falls to 0.28), so its `other` recall drops to 0.40. The lightweight demo
  model is exactly this n-gram, so its real-world rejection of unseen languages
  is weaker than the old number implied.
- **XLM-R (multilingual) rejects the unseen languages well** (`other` F1 0.96),
  so the **stacker is unaffected — in fact marginally higher (0.8811)** than the
  old leaky 0.8767, and now it is an *honest* number. So the leak had inflated
  the n-gram's apparent score, but the stacker was never relying on it.
- The shipped/committed `product_config` and `models/product_*` are left
  **unchanged**; `product_v2_*` are the corrected, leak-free artifacts.
- The `product_v2` XLM-R was early-stopped at epoch 3 on a validation plateau
  (macro-F1 0.8194 → 0.8653 → 0.8663); the epoch-3 checkpoint was promoted. See
  [VERIFIED_FACTS.md](VERIFIED_FACTS.md) §9.

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
  held-out Wikipedia test (ITDI-parity stacker): precision 0.80 / recall 0.51
  (under-fires; `product_v2` is similar at 0.74 / 0.67). On
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

Precision stays high (0.97-1.00) but recall collapses (0.24-0.47 depending on
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
  configs/   train/test splits: itdi_parity_*, product_v2_* (current),
             product_config_* (legacy, superseded — see VERIFIED_FACTS.md §9)
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
6. `split_product_v2.py` — rebuilds the product split with the `other` pool
   partitioned disjointly by Tatoeba language, so held-out `other` is a genuine
   unseen-language rejection test; `split_utils.assert_no_leak` runs on the
   final rows after every append and fails loudly on any train/test overlap
7. `model.py` — from-scratch char+word n-gram network, with optional
   balanced class weighting and per-class loss boosting
8. `transformer.py` — fine-tunes XLM-RoBERTa with the same split and
   weighting conventions
9. `stack_train.py` — logistic-regression meta-model over both base models'
   probabilities, meta-trained on the exact validation rows the base models
   held out (leakage-free by construction)
10. `predict.py` — CLI + importable inference, with a confidence floor for
   rejecting unrecognized input
11. `app.py` / `index.html` — Flask API + frontend; both derive their class
    list from the loaded model, not a hardcoded list, so the UI never falls
    out of sync when the class set changes
12. `itdi_eval.py` — eval-only scoring against the official VarDial 2022
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

**Note:** the Hub bundle holds the **original** `product_*` artifacts, trained
before the `other`-class leak was fixed. Their *training* is unaffected — the
leak invalidated the `other` class's held-out **score**, not the models
themselves — but the leak-free `product_v2_*` models score higher (stacker
0.8811 vs 0.8767) and are the ones to reproduce. Train them with Option B.
The 11-class **ITDI-parity** config is also not on the Hub.

### Option B — train from scratch

Train against the **leak-free `product_v2` split** (the committed
`product_config_*` files are legacy — kept only so the numbers in
[VERIFIED_FACTS.md](VERIFIED_FACTS.md) §1–8 stay reproducible). Regenerate the
split first if needed with `python scripts/split_product_v2.py`.

```bash
python scripts/model.py --data data/configs/product_v2_train.csv \
    --test data/configs/product_v2_test.csv --features both \
    --class-weights balanced --boost ladin=2.5 --boost ligurian=2.0 \
    --save models/product_v2_ngram_boost.pt

python scripts/transformer.py --data data/configs/product_v2_train.csv \
    --test data/configs/product_v2_test.csv --class-weights balanced \
    --boost ladin=2.5 --boost ligurian=2.0 --save models/product_v2_xlmr

python scripts/stack_train.py --data data/configs/product_v2_train.csv \
    --ngram models/product_v2_ngram_boost.pt --xlmr models/product_v2_xlmr \
    --test data/configs/product_v2_test.csv --save models/product_v2_stacker.joblib
```

Swap `product_v2` for `itdi_parity` throughout to train the 11-class config.

Then run the app against the models you just trained:

```bash
python scripts/app.py --model models/product_v2_stacker.joblib \
    --ngram models/product_v2_ngram_boost.pt --xlmr models/product_v2_xlmr \
    --floor 0.60
```

(The n-gram model alone — `--model models/product_v2_ngram_boost.pt`, no
`--ngram`/`--xlmr` — is a lightweight, fast alternative that doesn't need the
~1.1GB transformer loaded. Note its unseen-language rejection is materially
weaker than the stacker's: `other` F1 0.56 vs 0.96.)

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