"""
Build a cross-domain test set from Tatoeba
------------------------------------------
Tatoeba is a database of casual, conversational example sentences — a
completely different register from Wikipedia. Perfect for measuring how
your Wikipedia-trained model handles real, everyday dialect text.

Downloads one language's sentences and appends them to a CSV.

    python tatoeba.py --code scn --label sicilian   --out testset.csv
    python tatoeba.py --code nap --label neapolitan --out testset.csv
    python tatoeba.py --code vec --label venetian    --out testset.csv
    python tatoeba.py --code lmo --label lombard     --out testset.csv
    python tatoeba.py --code ita --label standard    --out testset.csv --max 300

Codes are ISO 639-3: scn, nap, vec, lmo, ita.
"""

import argparse
import bz2
import csv

import requests

from scraper import USER_AGENT, load_existing, write_rows

MIN_WORDS = 3      # keep short casual sentences, just drop 1-2 word fragments
MAX_WORDS = 60


def fetch_sentences(code):
    """Download + decompress one language's Tatoeba sentence export."""
    url = f"https://downloads.tatoeba.org/exports/per_language/{code}/{code}_sentences.tsv.bz2"
    print(f"Downloading {url} ...")
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    if r.status_code == 404:
        raise SystemExit(f"No Tatoeba export for '{code}'. Check the code, or that "
                         f"this dialect has sentences on Tatoeba.")
    r.raise_for_status()
    text = bz2.decompress(r.content).decode("utf-8")
    # each line: id <TAB> lang <TAB> sentence
    out = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out.append(parts[2].strip())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, help="Tatoeba/ISO 639-3 code: scn, nap, vec, lmo, ita")
    p.add_argument("--label", required=True, help="must match your training labels")
    p.add_argument("--out", default="testset.csv")
    p.add_argument("--max", type=int, default=None, help="cap sentences (useful for ita)")
    args = p.parse_args()

    sentences = fetch_sentences(args.code)
    good = [s for s in sentences if MIN_WORDS <= len(s.split()) <= MAX_WORDS]
    if args.max:
        good = good[:args.max]

    seen = load_existing(args.out)
    added = write_rows(args.out, good, args.label, f"tatoeba:{args.code}", seen)
    print(f"  {args.code}: {len(sentences)} raw -> {len(good)} usable -> "
          f"{added} new rows added to {args.out}")
    if len(good) < 20:
        print(f"  NOTE: only {len(good)} sentences — this dialect is thin on Tatoeba. "
              f"Still usable as a signal, just a small sample.")


if __name__ == "__main__":
    main()