"""
Italian Dialect Detector — Flask API server
==========================================
Serves the expanded classifier over a local web interface.

    pip install flask
    # n-gram model
    python app.py --model models/product_ngram_boost.pt --floor 0.60
    # best model (stacker)
    python app.py --model models/product_stacker.joblib \
        --ngram models/product_ngram_boost.pt --xlmr models/product_xlmr

Then open http://localhost:5000

THE CLASS LIST COMES FROM THE MODEL. It used to be read off DIALECT_META's
keys, which meant any class the model knew but the dict didn't was silently
dropped from the API response — the 7 dialects added in the expansion would
simply have never appeared in the UI, with no error anywhere. Now the response
is built from the model's own classes and DIALECT_META is only consulted for
display names, with a startup warning if it is missing an entry.

Works with both configs:
  * product (13 classes) — has a trained `other` rejection class, so a sentence
    can be rejected either by `other` outscoring the top dialect or by the
    confidence floor.
  * ITDI-parity (11 classes) — NO `other` class (that is what makes it
    benchmark-comparable), so the floor is the only rejection mechanism.
"""

import argparse
import os
import numpy as np
from flask import Flask, jsonify, request, send_file

from predict import Predictor

# index.html lives at the project root; this script lives in scripts/, one
# level down. Resolving relative to __file__ (not CWD) means `python app.py`
# works the same whether launched from the project root or from inside
# scripts/ — the plain send_file("index.html") broke the moment app.py moved
# into its own subfolder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(BASE_DIR, "..", "index.html")

app = Flask(__name__)

_pred = None
_floor = 0.60

# Display metadata. Non-dialect bookkeeping classes (`other`, `standard`) are
# marked so the UI can style them differently from the regional varieties.
# Colors were chosen per-dialect from real regional character (not a mechanical
# hue rotation) while keeping enough separation to read on a bar chart:
# cool blues/violets for the north, warm ambers/rust for the south, a genuinely
# distinct rose-mauve for Ladin since it's linguistically Rhaeto-Romance, not
# Italo-Romance, and shouldn't visually cluster with its Italo-Romance neighbors.
DIALECT_META = {
    "emilian":     {"it": "Emiliano-Romagnolo",  "region": "Emilia-Romagna · Nord Italia",
                    "color": "#A8763E"},
    "friulian":    {"it": "Friulano",            "region": "Friuli-Venezia Giulia · Nord-Est Italia",
                    "color": "#5C6B9E"},
    "ladin":       {"it": "Ladino",              "region": "Dolomiti · Trentino-Alto Adige",
                    "color": "#A8677A"},
    "ligurian":    {"it": "Ligure",              "region": "Liguria · Nord-Ovest Italia",
                    "color": "#2E9E9E"},
    "lombard":     {"it": "Lombardo",            "region": "Lombardia · Nord Italia",
                    "color": "#4A7CC7"},
    "neapolitan":  {"it": "Napoletano",          "region": "Campania · Sud Italia",
                    "color": "#D4622A"},
    "piedmontese": {"it": "Piemontese",          "region": "Piemonte · Nord-Ovest Italia",
                    "color": "#5B4B8A"},
    "sardinian":   {"it": "Sardo",               "region": "Sardegna",
                    "color": "#6E7C4F"},
    "sicilian":    {"it": "Siciliano",           "region": "Sicilia",
                    "color": "#C8981A"},
    "tarantino":   {"it": "Tarantino",           "region": "Taranto · Puglia",
                    "color": "#9E4A2E"},
    "venetian":    {"it": "Veneto",              "region": "Veneto · Nord-Est Italia",
                    "color": "#1F8A6E"},
    "standard":    {"it": "Italiano standard",   "region": "Italiano letterario",
                    "color": "#52525B", "is_dialect": False},
    "other":       {"it": "Non riconosciuto",    "region": "",
                    "color": "#BBBBBB", "is_dialect": False},
}


def _fallback_color(label):
    """Deterministic color for a class with no curated entry, so an unlisted
    class still renders distinctly instead of falling back to flat gray."""
    h = sum(ord(ch) for ch in label) % 360
    return f"hsl({h}, 45%, 45%)"


def meta_for(label):
    """Display metadata for a label, with a safe fallback so an unknown class
    still renders instead of disappearing from the response."""
    m = DIALECT_META.get(label)
    if m is None:
        return {"it": label.capitalize(), "region": "",
                "color": _fallback_color(label), "is_dialect": True}
    return {"it": m["it"], "region": m["region"],
            "color": m.get("color") or _fallback_color(label),
            "is_dialect": m.get("is_dialect", True)}


@app.route("/")
def index():
    return send_file(INDEX_HTML)


@app.route("/api/classes")
def classes():
    """Lets the frontend discover which classes this model actually serves."""
    return jsonify({
        "classes": _pred.canon,
        "meta": {c: meta_for(c) for c in _pred.canon},
        "has_other": "other" in _pred.canon,
        "floor": _floor,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    text = (request.get_json() or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400

    probs = _pred.proba([text])[0]
    scores = {c: float(probs[i]) for i, c in enumerate(_pred.canon)}

    # Build the ranked list from the MODEL's classes, not from DIALECT_META.
    # Only `other` is excluded — it is the rejection class, never a valid
    # answer. `standard` IS a valid, meaningful classification outcome ("this
    # is Standard Italian, not a regional dialect") and belongs in the ranking
    # and is eligible to be the top pick; is_dialect on it is display-only
    # metadata for the frontend to badge it differently, not a ranking filter.
    dialect_scores = sorted(
        [(c, s) for c, s in scores.items() if c != "other"],
        key=lambda x: -x[1]
    )
    if not dialect_scores:
        return jsonify({"error": "model exposes no dialect classes"}), 500

    top_label, top_score = dialect_scores[0]
    other_score = scores.get("other", 0.0)          # 0.0 for the ITDI-parity model
    standard_score = scores.get("standard", 0.0)

    # Rejection: an explicit `other` win (product config only) or a low top score.
    rejected = other_score > top_score or top_score < _floor

    return jsonify({
        "rejected": rejected,
        "top": top_label,
        "top_score": top_score,
        "other_score": other_score,
        "standard_score": standard_score,
        "has_other": "other" in _pred.canon,
        "scores": [{"label": l, "score": s} for l, s in dialect_scores],
        "meta": {c: meta_for(c) for c in _pred.canon},
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/product_ngram_boost.pt",
                   help="n-gram .pt or stacker .joblib — local path, or a filename "
                        "inside --hf-repo")
    p.add_argument("--ngram", default=None, help="base n-gram bundle (stacker mode) "
                   "— local path, or a filename inside --hf-repo")
    p.add_argument("--xlmr", default=None, help="base transformer dir (stacker mode) "
                   "— local path OR a Hugging Face Hub repo ID directly")
    p.add_argument("--hf-repo", default=None,
                   help="Hugging Face Hub repo ID to fetch --model/--ngram from. "
                        "Omit to use local paths (default).")
    p.add_argument("--floor", type=float, default=0.60)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address. Use 0.0.0.0 when running in a container "
                        "(Docker, Hugging Face Spaces) so it's reachable from "
                        "outside — 127.0.0.1 only accepts connections from "
                        "inside the same machine/container.")
    p.add_argument("--port", type=int, default=5000)
    a = p.parse_args()
    _floor = a.floor

    if not a.hf_repo and not os.path.exists(a.model):
        raise FileNotFoundError(
            f"Model not found: {a.model}\n"
            f"Train one with, e.g.:\n"
            f"  python scripts/model.py --data data/configs/product_config_train.csv "
            f"--test data/configs/product_config_test.csv --features both "
            f"--class-weights balanced --boost ladin=2.5 --boost ligurian=2.0 "
            f"--save models/product_ngram_boost.pt\n"
            f"...or pass --hf-repo to load a model already published to Hugging Face Hub.")

    _pred = Predictor(a.model, a.ngram, a.xlmr, hf_repo=a.hf_repo)

    missing = [c for c in _pred.canon if c not in DIALECT_META]
    if missing:
        print(f"  WARNING: no display metadata for {missing} — "
              f"they will render with fallback names. Add them to DIALECT_META.")

    print(f"\n  model   : {a.model} ({_pred.mode})"
          f"{' via ' + a.hf_repo if a.hf_repo else ''}")
    print(f"  classes : {len(_pred.canon)} -> {_pred.canon}")
    print(f"  reject  : {'`other` class + floor' if 'other' in _pred.canon else 'floor only (no `other` class)'}")
    print(f"  floor   : {_floor}")
    print(f"  open    : http://{a.host}:{a.port}\n")
    app.run(host=a.host, port=a.port, debug=False)