"""Italian Dialect Scraper (v3.2 — Wikipedia + Wikisource incl. multilingual host)"""
import argparse, csv, os, re, random, time
from urllib.parse import urlparse, unquote
import requests
try:
    import pysbd; _HAS_PYSBD = True
except ImportError:
    _HAS_PYSBD = False

CSV_PATH = "dataset.csv"
MIN_WORDS, MAX_SENT_WORDS, MAX_KEEP_WORDS = 5, 60, 100
USER_AGENT = "ItalianDialectScraper/3.2 (educational NLP project; contact: danche.j.1018@gmail.com)"
REQUEST_DELAY, MAX_BACKOFF = 0.8, 60
PROTECTED = ["a.C.", "d.C.", "A.C.", "D.C.", "sec.", "Sec.", "art.", "pag.", "ecc."]
SENT_BOUNDARY = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÀ-ÿ"\'«])')
PROJECTS = {"wikipedia", "wikisource", "wiktionary", "wikiquote", "wikibooks"}
SESSION = requests.Session()


def polite_get(url, params, max_retries=6):
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries):
        try:
            resp = SESSION.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            wait = min(MAX_BACKOFF, 2 ** attempt) + random.uniform(0, 1)
            print(f"    network error ({type(e).__name__}); waiting {wait:.0f}s ..."); time.sleep(wait); continue
        if resp.status_code in (429, 503):
            wait = int(resp.headers.get("Retry-After", 0)) or min(MAX_BACKOFF, 2 ** attempt)
            wait += random.uniform(0, 1)
            print(f"    rate-limited ({resp.status_code}); waiting {wait:.0f}s ..."); time.sleep(wait); continue
        if resp.status_code >= 400:
            resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            wait = min(MAX_BACKOFF, 2 ** attempt) + random.uniform(0, 1)
            print(f"    non-JSON response; waiting {wait:.0f}s ..."); time.sleep(wait); continue
        time.sleep(REQUEST_DELAY); return data
    raise RuntimeError(f"Gave up after {max_retries} retries: {url}")


def parse_wiki_url(url):
    parsed = urlparse(url); parts = parsed.netloc.split(".")
    if parts[0] in PROJECTS:
        lang, project = None, parts[0]
    else:
        lang = parts[0]; project = parts[1] if len(parts) > 2 else "wikipedia"
    return lang, unquote(parsed.path.split("/wiki/")[-1]), project


def fetch_extract(lang, title, project="wikipedia"):
    host = f"{lang}.{project}.org" if lang else f"{project}.org"
    params = {"action": "query", "titles": title, "prop": "extracts",
              "explaintext": "true", "redirects": "1", "format": "json"}
    data = polite_get(f"https://{host}/w/api.php", params)
    page = list(data["query"]["pages"].values())[0]
    if "missing" in page or "extract" not in page or not page["extract"].strip():
        return None
    return page["extract"]


def clean_text(raw):
    kept = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or re.match(r"^=+.*=+$", line) or len(line.split()) < MIN_WORDS:
            continue
        kept.append(line)
    return " ".join(kept)


def regex_split(text):
    safe = text
    for abbr in PROTECTED:
        safe = safe.replace(abbr, abbr.replace(".", "\x00"))
    return [p.replace("\x00", ".").strip() for p in SENT_BOUNDARY.split(safe) if p.strip()]


def split_sentences(text, lang_code="it"):
    raw = pysbd.Segmenter(language=lang_code, clean=True).segment(text) if _HAS_PYSBD else regex_split(text)
    out = []
    for s in raw:
        s = s.strip()
        if not s: continue
        out.extend(regex_split(s)) if len(s.split()) > MAX_SENT_WORDS else out.append(s)
    return out


def filter_sentences(sentences):
    return [s.strip() for s in sentences if MIN_WORDS <= len(s.split()) <= MAX_KEEP_WORDS]


def load_existing(csv_path):
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8", newline="") as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if row: seen.add(row[0])
    return seen


def write_rows(csv_path, sentences, label, source, seen):
    exists = os.path.exists(csv_path); added = 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["sentence", "label", "source"])
        for s in sentences:
            if s in seen: continue
            w.writerow([s, label, source]); seen.add(s); added += 1
    return added


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--label", required=True)
    p.add_argument("--out", default=CSV_PATH)
    a = p.parse_args()
    lang, title, project = parse_wiki_url(a.url)
    print(f"Fetching '{title}' from {(lang+'.' if lang else '')}{project}.org ...")
    raw = fetch_extract(lang, title, project=project)
    if raw is None: print("  !! Page not found — skipping."); return
    sents = filter_sentences(split_sentences(clean_text(raw)))
    seen = load_existing(a.out); added = write_rows(a.out, sents, a.label, title, seen)
    print(f"  Got {len(sents)} sentences, added {added} new.")


if __name__ == "__main__":
    main()