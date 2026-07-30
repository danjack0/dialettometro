# Data Sources

## Training data

### Wikipedia (dialect text)

Scraped fresh via `scripts/harvest_titles.py` + `scripts/scraper.py`, using
the MediaWiki API against each dialect's Wikipedia edition. No dump files are
redistributed in this repo — only the derived, cleaned sentence-level dataset
(`data/cleaned/balanced_13class.csv`) is committed.

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
Used for two purposes: mixing casual-register sentences into dialect classes
where coverage exists, and building the `other` rejection class (non-Italian
sentences across many languages) for the product config.

## Evaluation-only data (never trained on, never redistributed)

### VarDial 2022 ITDI shared task

[github.com/noe-eva/ITDI_2022](https://github.com/noe-eva/ITDI_2022)

Used via `scripts/itdi_eval.py` **strictly as a held-out external
benchmark**: the trained model is scored against their `dev.txt` and
`test_gold_standard.txt`, and nothing from this benchmark is fed back into
training or committed to this repo.

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