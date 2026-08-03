# VERIFIED_FACTS.md

Audit of `README.md` / `SOURCES.md` claims against what the code, data, and
saved model artifacts actually show. Every row is: **claim → verified value →
source command**. No models were retrained; every number below comes either
from reading the committed data or from running inference against already-saved
models (`scripts/itdi_eval.py`, the `Predictor` in `scripts/predict.py`).

- Date of audit: 2026-08-03
- Environment: Python 3.10.11, pandas 2.3.3, scikit-learn 1.7.2, torch 2.6.0+cu124, CUDA on (RTX 3060)
- ITDI benchmark located at sibling `../ITDI_2022` (outside this repo, as documented)

Legend: ✅ verified as stated · ⚠️ verified but claim needs correction/clarification · ❌ claim is wrong · ℹ️ context

---

## 1. Row counts

### `data/cleaned/balanced_13class.csv`

| Claim | Verified value | Source command |
|---|---|---|
| "~36,000 rows" | **35,967 rows** (≈36,000) ✅ | `python -c "import pandas as pd; print(len(pd.read_csv('data/cleaned/balanced_13class.csv')))"` |
| Named "13class" | **12 labels, not 13** ⚠️ — 11 dialects + `standard`; the 13th class (`other`) is **not** in this file, it is appended at split time | `pd.read_csv('data/cleaned/balanced_13class.csv')['label'].value_counts()` |

Per-class (balanced_13class.csv):

```
emilian 548 · friulian 4000 · ladin 1566 · ligurian 2417 · lombard 4000 ·
neapolitan 3930 · piedmontese 1648 · sardinian 4000 · sicilian 2938 ·
standard 4000 · tarantino 4000 · venetian 2920
```

Command: `python -c "import pandas as pd; print(pd.read_csv('data/cleaned/balanced_13class.csv')['label'].value_counts().sort_index())"`

### Split configs (`data/configs/`)

| Config | Total | Train | Test | Classes | Source command |
|---|---|---|---|---|---|
| itdi_parity | **31,967** | 25,576 | 6,391 | 11 (no standard, no other) | `python -c "import pandas as pd; print(len(pd.read_csv('data/configs/itdi_parity_train.csv')), len(pd.read_csv('data/configs/itdi_parity_test.csv')))"` |
| product | **48,265** | 40,574 | 7,691 | 13 (adds standard + other) | `python -c "import pandas as pd; print(len(pd.read_csv('data/configs/product_config_train.csv')), len(pd.read_csv('data/configs/product_config_test.csv')))"` |

Per-class **train / test**:

- **itdi_parity**: emilian 439/109 · friulian 3200/800 · ladin 1253/313 · ligurian 1934/483 · lombard 3200/800 · neapolitan 3144/786 · piedmontese 1319/329 · sardinian 3200/800 · sicilian 2351/587 · tarantino 3200/800 · venetian 2336/584
- **product**: same 11 dialects as itdi_parity, plus `standard` 3200/800 and `other` 11798/500

Consistency check (✅): itdi_parity total (31,967) = balanced_13class (35,967) − `standard` (4,000). Product total (48,265) = itdi_parity (31,967) + `standard` (4,000) + `other` (12,298).

Command: `python -c "import pandas as pd; [print(f, pd.read_csv(f)['label'].value_counts().sort_index()) for f in ['data/configs/itdi_parity_train.csv','data/configs/itdi_parity_test.csv','data/configs/product_config_train.csv','data/configs/product_config_test.csv']]"`

---

## 2. Distinct source wikis / Tatoeba sources

The committed CSVs' `source` column holds **article titles** (7,889 distinct in
balanced_13class), used as the group key for the article-level split — it is
not a wiki code. The wiki edition per row is implied by `label`. Provenance was
therefore verified from `data/titles/` (one harvest list per wiki) and from the
Tatoeba `source` tags in the product config.

| Claim (resume) | Verified value | Source command |
|---|---|---|
| "11 Wikipedia editions" | ⚠️ **Undercount for the full dataset.** 11 is correct for the *dialect* wikis / the 11-class ITDI-parity config. The full 13-class dataset draws from **12 Wikipedia editions** (adds `it` for Standard Italian) **plus Tatoeba**. | `ls data/titles/ | sed -E 's/_urls2?\.txt//' | sort -u` → 12 codes: eml, fur, it, lij, lld, lmo, nap, pms, sc, scn, tara, vec |
| Tatoeba sources | **34 distinct Tatoeba language subsets** ✅ (matches SOURCES.md's two-purpose description) | see below |

- **30 Tatoeba languages** feed the `other` rejection class (eng, deu, fra, spa, por, cat, ron, lat, nld, swe, nob, fin, pol, ces, vie, ...).
- **4 Tatoeba dialect ISO subsets** are mixed into dialect classes for casual register: `egl`→emilian, `fur`→friulian, `lij`→ligurian, `pms`→piedmontese.

Command: `python -c "import pandas as pd; d=pd.read_csv('data/configs/product_config_train.csv'); t=d[d['source'].astype(str).str.startswith('tatoeba:')]; print(t['source'].nunique()); print(t.groupby('label')['source'].nunique())"`

---

## 3. Held-out macro-F1 (own Wikipedia split, article-level)

Scored by running each saved model's `Predictor` against the committed test
split (inference only). "n-gram baseline" in the README = the **boosted**
n-gram (`*_ngram_boost.pt`), i.e. the exact model fed to the stacker; the plain
`*_ngram.pt` gives a different number (noted for completeness).

Reproduce (scratchpad scorer used during audit; equivalent to looping
`Predictor(...).proba` over the test CSV and taking argmax + `sklearn.f1_score(average='macro')`):
`python scripts/predict.py` internals via `from predict import Predictor`.

### ITDI-parity (11-class), test = `data/configs/itdi_parity_test.csv` (6,391 rows)

| Model | README | Verified macro-F1 |
|---|---|---|
| n-gram baseline (= `itdi_ngram_boost.pt`) | 0.8331 | **0.8331** ✅ |
| XLM-R (`itdi_xlmr`) | 0.8175 | **0.8175** ✅ |
| Stacker (`itdi_stacker.joblib`) | 0.8699 | **0.8699** ✅ |
| ℹ️ plain n-gram (`itdi_ngram.pt`), not in README | — | 0.8130 |

### Product (13-class), test = `data/configs/product_config_test.csv` (7,691 rows)

| Model | README | Verified macro-F1 |
|---|---|---|
| n-gram baseline (= `product_ngram_boost.pt`) | 0.8411 | **0.8411** ✅ |
| XLM-R (`product_xlmr`) | 0.8507 | **0.8507** ✅ |
| Stacker (`product_stacker.joblib`) | 0.8767 | **0.8767** ✅ |
| ℹ️ plain n-gram (`product_ngram.pt`), not in README | — | 0.8772 |

ℹ️ Note: the README's "the stacker beats both base models in both configs" holds
against the **boosted** base models it actually stacks. The *unboosted* product
n-gram (0.8772) marginally edges the product stacker (0.8767) on overall
macro-F1 — boosting Ligurian/Ladin weights trades overall macro-F1 for recall
on those classes. Not a table error; the README table reports the boosted model.

Command (per config): `python <scorer> --config {itdi|product}` where the
scorer loads `Predictor(model, ngram_boost, xlmr)`, batches `.proba()` over the
test `sentence` column, argmaxes to `canon`, and calls
`sklearn.metrics.f1_score(gold, pred, average='macro')`.

---

## 4. Per-class precision/recall/F1 — Ligurian, Tarantino, Ladin (internal held-out)

### ITDI-parity stacker (`itdi_stacker.joblib`)

| Class | P | R | F1 | support |
|---|---|---|---|---|
| ligurian | 0.8349 | 0.7329 | 0.7806 | 483 |
| tarantino | 0.9720 | 0.9537 | 0.9628 | 800 |
| ladin | 0.9924 | 0.4185 | 0.5888 | 313 |

### Product stacker (`product_stacker.joblib`)

| Class | P | R | F1 | support |
|---|---|---|---|---|
| ligurian | 0.8645 | 0.6998 | 0.7735 | 483 |
| tarantino | 0.9685 | 0.9613 | 0.9649 | 800 |
| ladin | 0.9841 | 0.3962 | 0.5649 | 313 |

Base-model per-class (for cross-referencing README prose):

| Model | Ligurian P/R/F1 | Tarantino F1 | Ladin P/R/F1 |
|---|---|---|---|
| itdi n-gram plain | 0.828/0.507/0.629 | 0.939 | 1.000/0.240/0.387 |
| itdi n-gram boost | 0.790/0.538/0.640 | 0.935 | 0.974/0.364/0.530 |
| itdi XLM-R | 0.931/0.505/0.655 | 0.943 | 0.990/0.320/0.483 |
| product n-gram plain | 0.974/0.768/0.859 | 0.957 | 0.993/0.422/0.592 |
| **product n-gram boost** | **0.804/0.509**/0.624 | 0.926 | 1.000/0.355/0.524 |
| product XLM-R | 0.964/0.551/0.701 | 0.947 | 0.991/0.342/0.508 |

Cross-check of README prose (all ✅):

- "Ligurian ... jumps from ~0.63 F1 (either base model alone) to ~0.78 (stacker)" ✅ (ITDI base 0.63–0.66 → stacker 0.78).
- "On the model's own held-out Wikipedia test: precision 0.80 / recall 0.51 (under-fires)" ✅ — **exactly** the product boosted n-gram Ligurian (P=0.8039, R=0.5093).
- Ladin "Precision stays high (0.97–1.00)" ✅ (0.974–1.000) · "recall collapses (0.24–0.42 depending on config)" ✅ (0.240 plain-itdi … 0.422 plain-product) · "~1,250 training rows" ✅ (ladin train = 1,253 both configs).
- Tarantino "F1 0.94–0.96 on the model's own held-out set" ✅ mostly (stacker 0.9628/0.9649; the product *boosted n-gram* is 0.926, slightly under 0.94 — the ~0.94–0.96 band describes the stacker/deployed model).

Command: same scorer as §3, reading `classification_report(gold, pred, output_dict=True)`.

---

## 5. External benchmark: `scripts/itdi_eval.py` vs VarDial dev & test (stacker)

Command:
```
python scripts/itdi_eval.py --model models/itdi_stacker.joblib \
  --ngram models/itdi_ngram_boost.pt --xlmr models/itdi_xlmr \
  --gold ../ITDI_2022/task/dev.txt   --pred-out <scratchpad>/dev_preds.txt
python scripts/itdi_eval.py --model models/itdi_stacker.joblib \
  --ngram models/itdi_ngram_boost.pt --xlmr models/itdi_xlmr \
  --gold ../ITDI_2022/task/test_gold_standard.txt --pred-out <scratchpad>/test_preds.txt
```
(Prediction files were written to the scratchpad, never the repo — they embed ITDI source text.)

| File | README weighted-F1 | Verified | README macro-F1 (present) | Verified | classes present | acc |
|---|---|---|---|---|---|---|
| dev.txt (6,799 rows) | 0.7865 | **0.7865** ✅ | 0.7928 | **0.7928** ✅ | 7: FUR, LIJ, LMO, PMS, SC, SCN, VEC ✅ | 0.7728 |
| test_gold_standard.txt (11,087 rows) | 0.5707 | **0.5707** ✅ | 0.5564 | **0.5564** ✅ | 8: EML, FUR, LIJ, LLD, LMO, NAP, ROA_TARA, VEC ✅ | 0.5747 |

Prose cross-checks (all ✅):

- "On test, it lands between the SVM-unigram and SVM-char-n-gram baselines" ✅ (0.4899 < 0.5707 < 0.7726).
- "On dev, the model ties the organizers' strongest baseline" ⚠️ roughly — dev weighted-F1 0.7865 vs the char-n-gram baseline 0.7726, but that baseline is a **test-set** number (see §6); there is no official dev baseline, so this is a cross-split comparison.
- Ligurian on ITDI "precision 0.46–0.53 / recall 0.75–0.95 (over-fires)" ✅ — dev LIJ P=0.4555/R=0.9530, test LIJ P=0.5279/R=0.7472. Confusion matrices confirm it absorbs Venetian (dev VEC→LIJ 367; test VEC→LIJ 276) and Sicilian (dev SCN→LIJ 211).
- Tarantino "0.21 on ITDI" ✅ — test ROA_TARA F1 = 0.2142; scatters to Neapolitan (184) and Sicilian (143) more than caught correctly (80).

---

## 6. Official baseline numbers → VarDial 2022 paper

The README/SOURCES baseline numbers trace **directly to the paper** and to the
organizers' own committed baseline outputs.

**Paper:** Aepli, Anastasopoulos, Chifu, Domingues, Faisal, Găman, Ionescu,
Scherrer. *Findings of the VarDial Evaluation Campaign 2022.* Proc. 9th VarDial
Workshop, COLING 2022. https://aclanthology.org/2022.vardial-1.1/ — §4.2
"Baselines" and **Table 3** (test-set ranking).

| Baseline | README (weighted-F1) | Paper Table 3 (weighted-F1 / macro-F1) | Verified |
|---|---|---|---|
| fastText (Baseline 1) | 0.1322 | **0.1322** / 0.1004 | ✅ |
| SVM unigram (Baseline 2) | 0.4899 | **0.4899** / 0.3424 | ✅ |
| SVM char n-gram (Baseline 3, best) | 0.7726 | **0.7726** / 0.5193 | ✅ |

Verbatim from the paper: *"We created three baselines. The weakest one
(Baseline 1) with a weighted F1-score of 0.1322 ... FastText."* · *"It resulted
in a weighted F1-score of 0.4899 ..."* · *"It reached a weighted F1-score of
0.7726 and was only outperformed by the three submissions of team SUKI."* ·
*"The submissions were ranked according to the weighted average F1-score"*
(confirming weighted-F1 as the shared-task headline metric).

These same numbers also reproduce from `../ITDI_2022/baselines/eval_baseline*.txt`
(the organizers' committed outputs). ⚠️ **All three baselines are
`test_gold_standard.txt` numbers** (support 11,087, 8 classes) — there is **no
official dev baseline** (see the correction in §7).

Command: `python -c "import fitz; d=fitz.open('<paper>.pdf'); print('\n'.join(p.get_text() for p in d))" | grep -nE "0.1322|0.4899|0.7726|weighted"` and `cat ../ITDI_2022/baselines/eval_baseline*.txt`

---

## 7. Corrections the README/SOURCES need (verified issues)

| # | Issue | Evidence | Command |
|---|---|---|---|
| A | ⚠️ `balanced_13class.csv` has **12 labels, not 13** (`other` is not in it). Filename is a misnomer. | §1 | `pd.read_csv('data/cleaned/balanced_13class.csv')['label'].nunique()` → 12 |
| B | ⚠️ "11 Wikipedia editions" undercounts the full dataset → **12 wikis (adds `it`) + Tatoeba**. | §2 | `ls data/titles/` |
| C | ❌ README lists the official baselines under **both** the dev and test columns, but they are **test-only** (no official dev baseline exists). | §6 | `cat ../ITDI_2022/baselines/eval_baseline*.txt` (files named `*_test_*`, support 11,087) |
| D | ❌ **Product `other` class is not held out.** All **500** test-`other` sentences also appear in train-`other`; the eval set is a subset of the train pool. `split_configs.py` appends `other_data.csv` whole to train and `other_eval.csv` whole to test (eval ⊂ data), and `leak_check` never inspects the appended `other` rows. Effect: `other` held-out metrics are invalid (R=0.998), and product macro-F1 is inflated by **≈+0.006** (0.8767 with `other` vs **0.8708** over the 12 non-`other` classes). Wikipedia dialect classes leak **zero** (article-level split is clean). | | `python -c "import pandas as pd; a=pd.read_csv('data/configs/product_config_train.csv'); b=pd.read_csv('data/configs/product_config_test.csv'); ao=set(a[a.label=='other'].sentence.astype(str)); bo=b[b.label=='other']; print((bo.sentence.astype(str).isin(ao)).sum(),'/',len(bo))"` → `500 / 500` |
| E | ❌ SOURCES.md: "only the derived, cleaned ... `balanced_13class.csv` is committed ... fetched fresh ... rather than redistributed" is **false**. 16 derived sentence-level CSVs (47.66 MB of Wikipedia+Tatoeba text) are committed, incl. `data/raw/combined_raw.csv` (88,244 rows), all config splits, `other_data.csv`, `other_eval.csv`, `data/cleaned/{flagged,removed_expansion}.csv`, legacy `balanced_*`. Raw XML **dumps** are indeed not committed (that part is true). **Resolved:** SOURCES.md now carries a full committed-data inventory. | | `git ls-files data \| grep '\.csv$'` |
| F | ⚠️ `LICENSE` is plain MIT over "the Software" with no data carve-out, but committed data is CC-BY-SA 4.0 (Wikipedia) / CC-BY 2.0 FR (Tatoeba). Needs code-vs-data scoping + attribution/share-alike note. | `LICENSE` | `cat LICENSE` |
| G | ✅ **RESOLVED.** 80 distinct sentences byte-identical to ITDI *dev*-set entries (79 SCN, 1 LMO; 0 from test) were found in 4 unused legacy files: `data/raw/testset_{train,eval}.csv`, `data/testset_{train,eval}_clean.csv`. Provenance was subsequently established as an **independent Sicilian Wikisource scrape**, not derivation from ITDI — every row carries its originating work in the `source` column (`Cappiddazzu paga tuttu/Atto I-III`, `'a vilanza/Atto I-III`, `Centona/La 'atta`), all public-domain literary texts ITDI holds no rights over; the ITDI files ship no provenance at all. The 4 files were from the early 5–6-class phase, never fed the current pipeline or the shipped models, and have been **removed** (`git rm`). Documented in SOURCES.md § Note on removed legacy files. | resolved this pass | `pd.read_csv(f)['source'].value_counts()` on the legacy files; `git ls-files data \| grep testset` → empty |

---

## 8. Claims verified TRUE (no change needed)

| Claim | Verified | Command / evidence |
|---|---|---|
| ✅ ITDI benchmark data is **never redistributed** | **0** ITDI dev/test sentences appear in any tracked file at HEAD; `itdi_*_preds.txt` gitignored (untracked); `models/` gitignored. (4 legacy files previously held 80 ITDI *dev*-overlapping sentences from an independent Wikisource scrape — see issue G — and were removed.) | `git ls-files \| grep -i pred` (only `scripts/predict.py`); field-level sentence-overlap scan over all tracked CSVs → 0 |
| ✅ "strict article-level splitting so no source document leaks" (Wikipedia) | 0 `(label, source)` groups shared across train/test for all Wikipedia classes, both configs | `(label,source)` intersection check → 0 wiki groups (30 shared groups are all `other||tatoeba:*`, i.e. the row-level `other` split, see issue D) |
| ✅ 11 dialects covered | emilian, friulian, ladin, ligurian, lombard, neapolitan, piedmontese, sardinian, sicilian, tarantino, venetian | label set of balanced_13class minus `standard` |
| ✅ char+word n-gram network | vectorizer = `FeatureUnion([char_wb (2,5) max 5000; word (1,2) max 5000])` = 10,000 dims → 2-layer MLP `DialectNet` | `torch.load('models/product_ngram_boost.pt')['vectorizer']` |
| ✅ logistic-regression meta-model / 3-way stacker | stacker `.joblib` holds `meta` (LogReg) over [n-gram probs ‖ XLM-R probs] + `canon` order | `joblib.load('models/*_stacker.joblib').keys()` → meta, canon, ngram_path, xlmr_path |
| ✅ "~1.1GB transformer" | `product_xlmr/model.safetensors` = 1,112,238,844 B ≈ 1.04 GB | `ls -l models/product_xlmr/model.safetensors` |
| ✅ dev 7 classes / test 8 classes | dev 6,799 rows / 7 classes; test 11,087 rows / 8 classes | line/label count of ITDI files |
| ✅ Live demo works | `https://dialettometro.onrender.com` → HTTP 200 (after ~21s Render cold start); `/api/classes` serves the 13-class product config (`has_other:true`, floor 0.6). Deploy reqs exclude transformers ⇒ it runs the n-gram product model. | `curl -sL https://dialettometro.onrender.com/api/classes` |