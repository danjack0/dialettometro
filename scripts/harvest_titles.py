"""
Wiki title harvester
=====================
Pulls real article titles from a wiki's MediaWiki API (list=allpages, main
namespace only) and writes them as full wiki:// URLs, one per line, ready to
feed into scraper.py in a loop.

Skips short/likely-stub-only titles is NOT done here on purpose — that
filtering belongs to audit_junk.py's Pass 2 (template detection), which needs
to see the actual scraped SENTENCES, not just titles, to tell a real article
from a bot stub. This script's only job is: give me N candidate titles.

Usage:
    python harvest_titles.py --wiki pms --project wikipedia --n 800 --out pms_urls.txt
    python harvest_titles.py --wiki roa-tara --project wikipedia --n 800 --out tara_urls.txt

--wiki is the subdomain code (pms, sc, lij, eml, roa-tara, lld, fur, ...).
--n is how many titles to fetch (allpages paginates in batches of up to 500).
"""

import argparse
import time
import random
import requests

USER_AGENT = "ItalianDialectScraper/3.2 (educational NLP project; contact: danche.j.1018@gmail.com)"

ap = argparse.ArgumentParser()
ap.add_argument("--wiki", required=True, help="wiki subdomain code, e.g. pms, sc, lij, eml, roa-tara, lld, fur")
ap.add_argument("--project", default="wikipedia")
ap.add_argument("--n", type=int, default=800, help="target number of titles")
ap.add_argument("--out", required=True)
ap.add_argument("--min-size", type=int, default=300,
                 help="skip pages with less than this many bytes of wikitext "
                      "source (default 300) — filters obvious stubs before "
                      "any HTTP round-trip to scraper.py")
args = ap.parse_args()

host = f"{args.wiki}.{args.project}.org"
api = f"https://{host}/w/api.php"
session = requests.Session()

titles = []
apcontinue = None
while len(titles) < args.n:
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,             # main article namespace only
        "apfilterredir": "nonredirects",  # exclude redirects — they have NO
                                           # extract text and were the actual
                                           # cause of the mass "not found" run
        "apminsize": args.min_size,   # skip near-empty stub pages up front
        "aplimit": 500,
        "format": "json",
    }
    if apcontinue:
        params["apcontinue"] = apcontinue

    for attempt in range(6):
        try:
            resp = session.get(api, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"    network error ({type(e).__name__}); waiting {wait:.0f}s ...")
            time.sleep(wait)
            continue
        if resp.status_code in (429, 503):
            wait = int(resp.headers.get("Retry-After", 0)) or min(60, 2 ** attempt)
            print(f"    rate-limited; waiting {wait:.0f}s ...")
            time.sleep(wait + random.uniform(0, 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        break
    else:
        raise RuntimeError(f"Gave up fetching allpages for {host}")

    batch = [p["title"] for p in data.get("query", {}).get("allpages", [])]
    if not batch:
        print(f"  No more pages returned for {host} — stopping with {len(titles)} titles.")
        break
    titles.extend(batch)
    print(f"  {host}: {len(titles)} titles so far ...")

    cont = data.get("continue", {})
    apcontinue = cont.get("apcontinue")
    if not apcontinue:
        print(f"  Reached end of {host}'s article list at {len(titles)} titles.")
        break
    time.sleep(0.5)

titles = titles[:args.n]

with open(args.out, "w", encoding="utf-8") as f:
    for t in titles:
        # urllib.parse.quote keeps this safe for spaces/special chars in titles
        from urllib.parse import quote
        f.write(f"https://{host}/wiki/{quote(t.replace(' ', '_'))}\n")

print(f"\nWrote {len(titles)} article URLs to {args.out}")
print(f"Next: while read url; do python scraper.py --url \"$url\" --label <LABEL> --out dataset_new.csv; done < {args.out}")