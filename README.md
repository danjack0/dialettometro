# Dialettometro — Italian Dialect Identifier

A 6-class Italian dialect classifier covering Sicilian, Venetian, Neapolitan,
Lombard, Standard Italian, and out-of-scope rejection. Ships as a local web app:
type a sentence, get a dialect and confidence score.

**Cross-domain macro-F1: 0.974** (stacked ensemble) / **0.96** (n-gram baseline)

---

## Key Finding

Fine-tuning XLM-RoBERTa (`xlm-roberta-base`) on this task matched but did not
beat a TF-IDF character + word n-gram baseline — macro-F1 0.96 on both, across
three independent rounds of data cleaning. This independently replicates the
central finding of the [VarDial 2022 ITDI shared task](https://aclanthology.org/2022.vardial-1.1/),
which found shallow n-gram models equal or outperform large pretrained transformers
on Italian dialect identification. The ceiling is the task — genuine lexical
overlap between geographically adjacent dialects — not model capacity.

A leakage-free stacked ensemble (logistic meta-learner over n-gram + XLM-R
probability vectors) raised the headline to **0.974**, confirming that the two
models make complementary errors (only 28% shared) even when neither individually
beats the other.

---

## Results

| Model | In-domain macro-F1 | Cross-domain macro-F1 |
|---|---|---|
| TF-IDF n-gram + MLP (baseline) | 0.93 | 0.96 |
| XLM-RoBERTa fine-tuned | 0.93 | 0.957 |
| **Stacked ensemble** | — | **0.974** |

*Cross-domain eval: `testset_eval_clean.csv` (2,392 sentences, casual register —
Tatoeba, Wikisource literary works). In-domain val: held-out Wikipedia articles,
source-aware split. Headline metric: macro-F1 (eval is ~42% Sicilian; raw
accuracy is misleading).*

Per-class cross-domain F1 (stacked ensemble, 6-class):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Lombard | 0.98 | 0.97 | 0.97 | 330 |
| Neapolitan | 0.97 | 0.95 | 0.96 | 171 |
| Sicilian | 0.99 | 0.96 | 0.97 | 1,000 |
| Standard Italian | 0.94 | 0.99 | 0.96 | 750 |
| Venetian | 0.92 | 0.92 | 0.92 | 141 |
| Other (rejection) | 0.98 | 0.95 | 0.96 | 500 |

---

## Methodology

### Data

All data sourced, audited, and licensed for public redistribution under
**CC-BY-SA 4.0** (see [SOURCES.md](SOURCES.md)).

| File | Rows | Description |
|---|---|---|
| `balanced_clean.csv` | 3,353 | 5-class Wikipedia training data, ~669/class |
| `balanced_6class.csv` | 4,044 | + 691 `other` sentences (22 Latin-script languages) |
| `testset_train_clean.csv` | 2,414 | Casual-register sentences mixed into training |
| `testset_eval_clean.csv` | 2,392 | **Held-out scoreboard** — never used in training |
| `other_data.csv` | ~11,000 | Latin-script language sentences for `other` class |

Schema: `sentence, label, source` — the `source` column records per-row
provenance and is used for the source-aware split.

### Integrity measures

- **Source-aware split:** `GroupShuffleSplit` on the `source` column ensures
  validation articles are fully held out — no sentence from the same Wikipedia
  article appears in both train and val. Prevents topic memorization.
- **Zero train/eval overlap:** verified programmatically. `testset_eval_clean.csv`
  was never seen by any model during training.
- **Data cleaning, blind to predictions:** 102 junk rows (ISBNs, Latin taxonomy,
  English metadata) and 25 confirmed mislabels removed by content rule applied
  uniformly — flagged rows were removed whether the model got them right or wrong.
  Manifests of every removed row are in `removed_*.csv`.
- **Stacker trained on held-out data:** the logistic meta-learner is trained on
  the 800-row in-domain Wikipedia val split (never seen by base model weights),
  then scored once on `testset_eval_clean.csv`.

### Pipeline

```
Wikipedia / Tatoeba / Wikisource
         │
    scraper.py / tatoeba.py          ← data collection
         │
    clean_data.py / prep_other.py    ← junk removal, mislabel removal, other-class prep
         │
    model.py                         ← TF-IDF + MLP baseline
    transformer.py                   ← XLM-RoBERTa fine-tuning
         │
    stack_train.py                   ← leakage-free stacked ensemble
         │
    app.py + index.html              ← local web interface
```

---

## Setup

**Prerequisites:** Python 3.10+, pip

```bash
git clone https://github.com/YOUR_USERNAME/dialettometro.git
cd dialettometro
pip install torch scikit-learn numpy transformers accelerate sentencepiece flask
```

**Train the baseline model** (CPU-friendly, ~2 min):

```bash
python model.py --data balanced_6class.csv --extra-train testset_train_clean.csv \
    --test testset_eval_clean.csv --features both --save dialect_ngram.pt
```

**Fine-tune the transformer** (GPU recommended, ~10 min on RTX 3060):

```bash
python transformer.py --data balanced_6class.csv --extra-train testset_train_clean.csv \
    --test testset_eval_clean.csv --batch 8 --grad-accum 2 --save dialect_xlmr
```

**Build the stacked ensemble** (requires both models saved above):

```bash
python stack_train.py --data balanced_6class.csv --ngram dialect_ngram.pt \
    --xlmr dialect_xlmr --test testset_eval_clean.csv --save stacker.joblib
```

---

## Usage

**Web interface** (recommended):

```bash
python app.py --model dialect_ngram.pt --floor 0.60
# open http://localhost:5000
```

**CLI — single sentence:**

```bash
python predict.py --model dialect_ngram.pt --floor 0.60 "Napule è 'a cchiù bella città"
# neapolitan  (94%)   [next: sicilian 3%, standard 2%]
```

**CLI — interactive:**

```bash
python predict.py --model dialect_ngram.pt --floor 0.60
```

**Tune the confidence floor** (prints rejection table for your eval set):

```bash
python predict.py --model dialect_ngram.pt --calibrate testset_eval_clean.csv
```

The `--floor` flag controls the confidence threshold below which input is returned
as *uncertain* rather than forced into a label. At 0.60: keeps 96% of real
dialect sentences, correctly rejects 51% of errors and most non-Italian-family
input. Non-Latin-script input (Cyrillic, Arabic, CJK) scores near-zero and is
caught automatically; Latin-script languages (English, French, etc.) are handled
by the explicit `other` class.

---

## Repository structure

```
dialettometro/
├── model.py              TF-IDF + MLP baseline
├── transformer.py        XLM-RoBERTa fine-tuning
├── ensemble.py           Error-overlap diagnostic
├── stack_train.py        Leakage-free stacked ensemble
├── predict.py            CLI inference with confidence floor
├── app.py                Flask API server
├── index.html            Web frontend (Dialettometro)
├── scraper.py            Wikipedia / Wikisource scraper
├── tatoeba.py            Tatoeba sentence downloader
├── audit_junk.py         Content-rule junk auditor
├── clean_data.py         Safe data cleaner with integrity checks
├── prep_other.py         other-class balancing for 6-class retrain
├── SOURCES.md            Full data provenance and licensing
└── data/
    ├── balanced_clean.csv
    ├── balanced_6class.csv
    ├── testset_train_clean.csv
    ├── testset_eval_clean.csv
    └── other_data.csv
```

Trained model files (`dialect_ngram.pt`, `dialect_xlmr/`, `stacker.joblib`) are
not included — generate them with the training commands above.

---

## Data sources & licensing

This dataset is derived from:

- **Wikipedia** (Italian, Neapolitan, Sicilian, Lombard, and Venetian editions),
  © Wikipedia contributors, licensed CC-BY-SA 4.0.
- **Tatoeba** (https://tatoeba.org), licensed CC-BY 2.0 FR.
- **Wikisource** public-domain texts by Nino Martoglio, Luigi Pirandello,
  Giovanni Boccaccio, and Ippolito Cavalcanti, plus an anonymous Bible-parable
  translation.

Per-sentence provenance is recorded in the `source` column of each CSV.
This dataset is released under **CC-BY-SA 4.0**. Code is released under the **MIT License**.

See [SOURCES.md](SOURCES.md) for full attribution, license obligations, and
documentation of two deliberately excluded sources.

---

## Limitations

- **Closed-world classifier:** covers five Italian dialect varieties plus an
  `other` rejection class. Sardinian, Piedmontese, Ligurian, and other Italian
  varieties not in scope are forced into the nearest covered class unless caught
  by the `other` model or the confidence floor.
- **Register gap:** training is Wikipedia-heavy (formal/encyclopedic). Casual,
  spoken, and social-media register performs somewhat lower, particularly for
  Venetian and Neapolitan — the two classes with the thinnest casual-data coverage.
- **Short text:** sentences under ~5 words are unreliable. The confidence floor
  mitigates this by returning *uncertain* on low-confidence predictions.
- **Dialect continuum:** Lombard↔Venetian and Venetian↔Neapolitan overlaps are
  linguistically real. No model eliminates these; they represent the hard limit
  of the task at this data scale.

---

## Related work

- Zampieri et al., *Findings of the VarDial Evaluation Campaign 2022*,
  ACL Anthology 2022.vardial-1.1
- Universal Dependencies Italian treebanks (standard Italian)
- ITDI 2022 shared task data: `noe-eva/ITDI_2022` (GitHub)
