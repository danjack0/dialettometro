"""
upload_to_hub.py — push trained models to Hugging Face Hub
============================================================
Run this ONCE per config, after training, to publish the model weights
somewhere that can actually hold them (GitHub hard-rejects files over 100MB;
Hub has no practical limit for public repos and is free).

Each config becomes ONE Hub repo containing everything needed to run it:
the full XLM-R directory (config.json, model.safetensors, tokenizer files)
PLUS the n-gram bundle (.pt) and stacker (.joblib) as extra files alongside.
One repo = one complete, downloadable config.

Before running: `huggingface-cli login` (see README for the one-time token
setup — you only do this once ever, not per upload).

Usage:
    python upload_to_hub.py --username YOUR_HF_USERNAME --config product
    python upload_to_hub.py --username YOUR_HF_USERNAME --config itdi

This does NOT touch your GitHub repo. Model weights live on the Hub;
code stays on GitHub; the two are connected only by the repo ID strings
you pass to --xlmr / --hf-repo at inference time.
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo

ap = argparse.ArgumentParser()
ap.add_argument("--username", required=True, help="your Hugging Face username")
ap.add_argument("--config", required=True, choices=["product", "itdi"])
ap.add_argument("--models-dir", default="models")
ap.add_argument("--repo-suffix", default=None,
                help="override the repo name suffix (default: same as --config)")
ap.add_argument("--private", action="store_true",
                help="make the Hub repo private (default: public, which is free "
                     "and unlimited storage; private has a much smaller free tier)")
args = ap.parse_args()

FILES = {
    "product": {
        "xlmr_dir": "product_xlmr",
        "ngram": "product_ngram_boost.pt",
        "stacker": "product_stacker.joblib",
    },
    "itdi": {
        "xlmr_dir": "itdi_xlmr",
        "ngram": "itdi_ngram_boost.pt",
        "stacker": "itdi_stacker.joblib",
    },
}[args.config]

suffix = args.repo_suffix or args.config
repo_id = f"{args.username}/dialettometro-{suffix}"

xlmr_path = os.path.join(args.models_dir, FILES["xlmr_dir"])
ngram_path = os.path.join(args.models_dir, FILES["ngram"])
stacker_path = os.path.join(args.models_dir, FILES["stacker"])

for p in (xlmr_path, ngram_path, stacker_path):
    if not os.path.exists(p):
        raise SystemExit(f"Not found: {p}\nRun this from the project root, "
                         f"or pass --models-dir.")

print(f"Creating (or reusing) Hub repo: {repo_id}  (private={args.private})")
create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

api = HfApi()

print(f"\nUploading {xlmr_path}/ (this is the big one, ~1.1GB — will take a "
      f"few minutes depending on your connection)...")
api.upload_folder(
    folder_path=xlmr_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload XLM-R weights",
)

print(f"\nUploading {ngram_path} ...")
api.upload_file(
    path_or_fileobj=ngram_path,
    path_in_repo=os.path.basename(ngram_path),
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload n-gram bundle",
)

print(f"\nUploading {stacker_path} ...")
api.upload_file(
    path_or_fileobj=stacker_path,
    path_in_repo=os.path.basename(stacker_path),
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload stacker",
)

print(f"\nDone. Repo: https://huggingface.co/{repo_id}")
print(f"\nTo run inference against it (no local model files needed):")
print(f"  python predict.py --hf-repo {repo_id} \\")
print(f"      --model {os.path.basename(stacker_path)} \\")
print(f"      --ngram {os.path.basename(ngram_path)} \\")
print(f"      --xlmr {repo_id} \"...\"")
