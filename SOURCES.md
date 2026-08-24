# Data Sources

## Training data

### Wikipedia (dialect text)

Scraped via `scripts/harvest_titles.py` + `scripts/scraper.py`, using the
MediaWiki API against each edition below. The data draws from **12 Wikipedia
editions**: the 11 dialect wikis plus `it` (Italian) for the Standard-Italian
class.

Raw Wikipedia XML **dumps are not redistributed** here — the scripts re-fetch
those from source. Derived, **sentence-level** CSVs, however, **are** committed:
the cleaned corpus (`data/cleaned/balanced_13class.csv`), the train/test splits
(`data/configs/`), and intermediate/scraped sentence files under `data/raw/`
(e.g. `combined_raw.csv`). This text is CC-BY-SA 4.0, so committing it is
permitted with attribution + share-alike (see the README's Data & Licensing
section). Note `balanced_13class.csv` itself holds 12 labels (11 dialects +
Standard); the 13th class, `other`, is Tatoeba-sourced and appended when the
product split is built.

| Dialect | Wiki code |
|---|---|
| Emilian-Romagnol | `eml` |
| Friulian | `fur` |
| Ladin | `lld` |
| Ligurian | `lij` |
| Lombard | `lmo` |
| Neapolitan | `nap` |
| Piedmontese | `pms` |
| Sardinian | `sc` |
| Sicilian | `scn` |
| Tarantino | `roa-tara` |
| Venetian | `vec` |
| Standard Italian | `it` |

Wikipedia text is licensed CC BY-SA 4.0. Article text was fetched via the
MediaWiki `action=query&prop=extracts` API, filtered to exclude redirects and
near-empty stubs at harvest time (`apfilterredir=nonredirects`,
`apminsize`).

### Tatoeba (casual-register + rejection-class text)

Fetched via `scripts/tatoeba.py` using the Tatoeba API, filtered by
ISO 639-3 code per dialect where available. Licensed CC BY 2.0 FR.
Used for two purposes (34 distinct Tatoeba language subsets in total):

- **Casual-register mixing** into dialect classes where coverage exists — 4
  subsets: `egl`→Emilian, `fur`→Friulian, `lij`→Ligurian, `pms`→Piedmontese.
- **The `other` rejection class** (product config only) — 30 non-Italian
  language subsets (e.g. eng, deu, fra, spa, por, cat, ron, lat, ...).

## Committed data inventory

Every CSV tracked under `data/` is listed below (18 files, **53.63 MB**; all
tracked files under `data/`, including the 15 `titles/*.txt` harvest lists,
total **54.80 MB** across 33 files). All text is Wikipedia-derived (CC BY-SA
4.0) or Tatoeba-derived (CC BY 2.0 FR). Reproduce this list with
`git ls-files data | grep '\.csv$'`.

**Active pipeline — cleaned corpus and final splits**

- `data/cleaned/balanced_13class.csv` — 35,967 rows — post-audit balanced
  corpus (12 labels: 11 dialects + Standard).
- `data/configs/itdi_parity_train.csv` / `itdi_parity_test.csv` — 25,576 /
  6,391 — the 11-class ITDI-parity split.
- `data/configs/product_v2_train.csv` / `product_v2_test.csv` — 38,274 /
  9,490 — the current, leak-free 13-class product split (adds Standard +
  `other`). The `other` class is partitioned **by Tatoeba language**, so 5 of
  the 30 languages are held out of training entirely and no sentence straddles
  train and test.
- `data/configs/product_config_train.csv` / `product_config_test.csv` —
  40,574 / 7,691 — the original 13-class product split. **Legacy —
  superseded by `product_v2`**: all 500 of its test-`other` rows also appear
  in training, so its `other` class has no valid held-out score. Retained
  because the shipped `models/product_*` (including the live demo's n-gram)
  were trained on it.

**Active pipeline — raw scrape (pre-cleaning inputs)**

- `data/raw/combined_raw.csv` — 88,244 rows — all scraped Wikipedia + Tatoeba
  sentences before cleaning/balancing.
- `data/raw/dataset_new.csv` (41,665), `data/raw/dataset_orig5.csv` (16,354),
  `data/raw/dataset_std.csv` (15,126 — Standard-Italian `it` scrape) — earlier
  per-batch scrape files retained from the pipeline's history.
- `data/raw/other_data.csv` (11,798) / `data/raw/other_eval.csv` (500) —
  the Tatoeba `other`-class pool and its held-out subset (product config).

**Audit transparency artifacts**

- `data/cleaned/flagged.csv` — 7,486 rows — rows flagged by `audit_junk.py`
  (with the rule that fired), kept for auditability.
- `data/cleaned/removed_expansion.csv` — 28,745 rows — rows dropped during
  expansion cleaning (with a `reason` column).

**Legacy / superseded (early 5–6-class experiments; NOT used by the current
pipeline or the shipped models)**

- `data/balanced_6class.csv` (4,044), `data/balanced_clean.csv` (3,353),
  `data/raw/balanced.csv` (3,455)

Four further legacy files — `data/raw/testset_{train,eval}.csv` and
`data/testset_{train,eval}_clean.csv` — were removed from the repo; see
[Note on removed legacy files](#note-on-removed-legacy-files) at the end of
this document.

## Evaluation-only data (never trained on, never redistributed)

### VarDial 2022 ITDI shared task

[github.com/noe-eva/ITDI_2022](https://github.com/noe-eva/ITDI_2022)

Used via `scripts/itdi_eval.py` **strictly as a held-out external
benchmark**: the trained model is scored against their `dev.txt` and
`test_gold_standard.txt`. Nothing from this benchmark is fed back into
training, and no ITDI file — or prediction derived from one — is committed to
this repo.

**Why eval-only, not training data:** the ITDI_2022 repository ships no
LICENSE file. Its dev/test files are described in their own documentation as
"newly collected text samples" whose sources were "unknown even to
participants" — unclear provenance plus no license grant means this data must
not be redistributed or baked into a released corpus. Scoring against it is
the use the shared task was designed for (that is the entire purpose of a
held-out test set); training on it, or committing their files or predictions
derived from them, is not.

Practical handling:
- Their repo is cloned outside this project's directory tree, never inside it
- `itdi_predictions.txt`, `itdi_test_preds.txt`, `itdi_dev_preds.txt`
  (prediction files that embed their source sentences) are excluded via
  `.gitignore`
- Only aggregate metrics (accuracy, macro-F1, weighted-F1, per-class
  precision/recall/F1) are reported — never per-sentence predictions
  alongside their original text
- Cited as: Aepli et al., VarDial 2022 ITDI shared task

Official baseline scores (`test_gold_standard.txt`), included in this repo's
README for context: fastText (weighted-F1 0.1322), SVM unigram (0.4899), SVM
character n-gram (0.7726, their strongest baseline).

## Note on removed legacy files

An early 5–6-class prototype scraped Sicilian **Wikisource** (public-domain
literary texts: Capuana's *Cappiddazzu paga tuttu*, Martoglio's *'A vilanza*
and *Centona*, and a Neapolitan proverb collection). Those files —
`data/raw/testset_{train,eval}.csv` and `data/testset_{train,eval}_clean.csv` —
were superseded by the Wikipedia-API pipeline and have been removed.

A byte-level audit found 80 sentences in them (79 Sicilian, 1 Lombard)
identical to entries in the VarDial ITDI dev set. This reflects **shared
upstream sources, not derivation from ITDI**: each row carries its originating
Wikisource work in the `source` column (e.g. `Cappiddazzu paga tuttu/Atto I`,
`'a vilanza/Atto II`, `Centona/La 'atta`), whereas the ITDI files ship no
provenance at all ("sources unknown even to participants"). The underlying
texts are public-domain literary works that ITDI holds no rights over.

These files never fed the current pipeline. `balanced_13class.csv`, both split
configs, `combined_raw.csv`, and every trained model are derived solely from
the MediaWiki API and Tatoeba, and contain **zero** ITDI-overlapping sentences.