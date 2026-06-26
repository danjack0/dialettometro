# Data Sources & Licensing

This document lists every source used to build the Italian dialect classifier
dataset, its license, and the attribution required to redistribute it. All
sources below were audited and are cleared for publication. Two candidate
sources were deliberately **excluded** to keep the dataset clean — they are
documented at the end for transparency.

> **Bottom line for redistribution:** the combined dataset contains CC-BY-SA
> material, so the aggregate must be released under **CC-BY-SA 4.0** (see
> "Combined-license obligation" below). Attribution is required for the
> Wikimedia and Tatoeba portions; the public-domain works carry no obligation
> but are credited here for provenance.

---

## 1. Wikipedia — training data and part of the test set

- **What:** Encyclopedic article text from the Italian, Neapolitan, Sicilian,
  Lombard, and Venetian Wikipedias (`it`, `nap`, `scn`, `lmo`, `vec`).
- **Files:** all of `dataset.csv` / `balanced.csv`; the "Wiki page title"
  rows in `testset.csv` / `testset_train.csv` / `testset_eval.csv`.
- **License:** Creative Commons Attribution-ShareAlike (CC-BY-SA 4.0).
- **Obligation:** attribution **and** share-alike. Article titles are stored
  in the `source` column of each row.

Includes two Neapolitan proverb/idiom pages confirmed to be on
`nap.wikipedia.org` (CC-BY-SA), not third-party compilations:
- `Pruverbie napulitane`
- `Lista 'e mode 'e dicere napuletane`

## 2. Tatoeba — casual-register test data

- **What:** Conversational example sentences in five language codes
  (`ita`, `nap`, `scn`, `lmo`, `vec`).
- **Files:** the `tatoeba:*` rows in `testset.csv` and its splits.
- **License:** CC-BY 2.0 FR.
- **Obligation:** attribution only. Per Tatoeba's own guidance, credit the
  source with a link to https://tatoeba.org and a mention of CC-BY 2.0 FR.
- **Commercial use:** permitted under CC-BY (Tatoeba's text corpus, not the
  separately-licensed audio).

## 3. Wikisource — public-domain literary works (test set only)

These are early-modern and early-20th-century literary texts whose copyright
has expired. Author death + 70 years governs in the EU; all are clear. Stored
in the `source` column under their work titles.

| Work | Author(s) | Status |
|------|-----------|--------|
| *Cappiddazzu paga tuttu* | Nino Martoglio (d. 1921) & Luigi Pirandello (d. 1936) | PD in EU since 2007 (joint work; term runs from last author, Pirandello); published 1917/1922 |
| *'A vilanza* | Nino Martoglio & Luigi Pirandello | PD, same basis |
| *Centona* (poetry) | Nino Martoglio (d. 1921) | PD since 1992 |
| *Scuru* | Nino Martoglio (d. 1921) | PD since 1992 |
| *San Giuvanni Decullatu* | Nino Martoglio (d. 1921) | PD since 1992 |
| *La parobola del Figliol Prodigo* | Bible-parable translation | PD |
| *Novella IX, Giornata I — Decamerone* (Lombard) | Giovanni Boccaccio (14th c.) | PD |
| *Cucina teorico-pratica* | Ippolito Cavalcanti (1837) | PD |

A small number of rows are tagged `Paggena prencepale` (Neapolitan Wikisource
main-page content) and `wikiquote`; these are CC-BY-SA Wikimedia content and
fall under the attribution + share-alike terms in §1.

> **Note:** the two 10-row works *Scuru* and *San Giuvanni Decullatu* are
> attributed to Martoglio by the same authorship basis as *Centona*. Each
> Wikisource page carries a license/PD tag at the bottom if you want a
> per-page confirmation.

---

## Combined-license obligation

Licenses in the dataset mix as follows:

- **CC-BY-SA 4.0** (Wikipedia) — share-alike: any redistributed dataset that
  includes this material must itself be CC-BY-SA.
- **CC-BY 2.0 FR** (Tatoeba) — attribution only; compatible with inclusion in
  a CC-BY-SA aggregate.
- **Public domain** (Wikisource works) — no restriction; compatible with
  anything.

Because share-alike is the most restrictive term present and it propagates to
the whole, **release the combined dataset under CC-BY-SA 4.0** with the
attributions above. (This is the standard reading of CC-BY-SA's share-alike
clause; it is not legal advice. If the stakes warrant it, confirm with a
source qualified to advise on licensing.)

---

## Suggested attribution block

> This dataset is derived from:
> - **Wikipedia** (Italian, Neapolitan, Sicilian, Lombard, and Venetian
>   editions), © Wikipedia contributors, licensed CC-BY-SA 4.0.
> - **Tatoeba** (https://tatoeba.org), licensed CC-BY 2.0 FR.
> - **Wikisource** public-domain texts by Nino Martoglio, Luigi Pirandello,
>   Giovanni Boccaccio, and Ippolito Cavalcanti, plus an anonymous Bible-parable
>   translation.
>
> Per-sentence provenance is recorded in the `source` column of each CSV.
> This dataset is released under CC-BY-SA 4.0.

---

## Excluded sources (documented for transparency)

These were considered and deliberately left out. Neither appears in any
published file.

- **Neapolitan-Spoken-Corpus** (Hugging Face: `anonymous-nsc-author/
  Neapolitan-Spoken-Corpus`) — licensed **CC-BY-NC-4.0**. Legally usable, but
  the **NonCommercial** term would force the entire combined dataset to NC and
  is incompatible with the commercial-friendly CC-BY/CC-BY-SA mix above.
  Excluded to keep the dataset commercially usable. (The play/poetry portion
  of this corpus also drew on works by 20th-century authors still in copyright,
  e.g. Eduardo De Filippo, d. 1984.)
- **`sicilian_dataset_raw.csv`** (NLLB-style machine-mined English↔Sicilian
  bitext) — excluded on **quality** grounds (machine-translation artifacts,
  translationese register) and because Sicilian was already the
  best-represented class. Not a licensing decision.
