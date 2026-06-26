"""
Italian Dialect Detector — Flask API server
==========================================
Loads dialect_ngram.pt and serves predictions via a local web interface.
Place this file in the same folder as dialect_ngram.pt and index.html.

    pip install flask
    python app.py --model dialect_ngram.pt --floor 0.60

Then open http://localhost:5000 in your browser.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ---- model ----
_model = None
_vec = None
_classes = None
_floor = 0.60

DIALECT_META = {
    "lombard":    {"it": "Lombardo",         "region": "Lombardia · Nord Italia"},
    "neapolitan": {"it": "Napoletano",        "region": "Campania · Sud Italia"},
    "sicilian":   {"it": "Siciliano",         "region": "Sicilia"},
    "standard":   {"it": "Italiano standard", "region": "Italiano letterario"},
    "venetian":   {"it": "Veneto",            "region": "Veneto · Nord-Est Italia"},
    "other":      {"it": "Non riconosciuto",  "region": ""},
}


class DialectNet(nn.Module):
    def __init__(self, d, h, n):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(h, n)
    def forward(self, x):
        return self.fc2(self.drop(self.relu(self.fc1(x))))


def load_model(path):
    global _model, _vec, _classes
    b = torch.load(path, map_location="cpu", weights_only=False)
    _classes = list(b["classes"])
    _model = DialectNet(b["input_dim"], b["hidden"], len(_classes))
    _model.load_state_dict(b["state_dict"])
    _model.eval()
    _vec = b["vectorizer"]


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    text = (request.get_json() or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400

    X = torch.from_numpy(_vec.transform([text]).toarray().astype(np.float32))
    with torch.no_grad():
        probs = torch.softmax(_model(X), 1)[0].numpy()

    scores = {c: float(probs[i]) for i, c in enumerate(_classes)}

    # dialect-only scores (excluding 'other'), sorted descending
    dialect_scores = sorted(
        [(c, scores.get(c, 0.0)) for c in DIALECT_META if c != "other"],
        key=lambda x: -x[1]
    )
    top_label, top_score = dialect_scores[0]
    other_score = scores.get("other", 0.0)

    rejected = other_score > top_score or top_score < _floor

    return jsonify({
        "rejected": rejected,
        "top": top_label,
        "top_score": top_score,
        "other_score": other_score,
        "scores": [{"label": l, "score": s} for l, s in dialect_scores],
        "meta": DIALECT_META,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dialect_ngram.pt")
    p.add_argument("--floor", type=float, default=0.60)
    p.add_argument("--port", type=int, default=5000)
    a = p.parse_args()
    _floor = a.floor

    if not os.path.exists(a.model):
        raise FileNotFoundError(
            f"Model not found: {a.model}\n"
            f"Train with: python model.py --data balanced_6class.csv "
            f"--extra-train testset_train_clean.csv --test testset_eval_clean.csv "
            f"--features both --save dialect_ngram.pt"
        )

    load_model(a.model)
    print(f"\n  model : {a.model}")
    print(f"  floor : {_floor}")
    print(f"  open  : http://localhost:{a.port}\n")
    app.run(port=a.port, debug=False)
