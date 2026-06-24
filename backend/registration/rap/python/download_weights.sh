#!/usr/bin/env bash
# Fetch the RAP pretrained checkpoints into ./weights/ (next to the worker).
#
# The weights are a RUNTIME concern only; they are never required to build
# CloudCropper and are .gitignored.
#
# Recon (docs/design/_rap-upstream-notes.md): two distribution channels.
#   HuggingFace YuePanEdward/RAP (MIT, ~686 MB total):
#     rap_model_10.ckpt   (~327 MB)  app.py "M (rap_10)", config rap_10
#     rap_model_12.ckpt   (~356 MB)  app.py "L (rap_12)", config rap_12 (default)
#     mini_spinnet_t.pth  (~3.67 MB) the SpinNet local-feature extractor
#   OR ipb.uni-bonn.de (upstream scripts/download_weights_and_demo_data.sh):
#     wget https://www.ipb.uni-bonn.de/html/projects/rap/weights.zip ; unzip ; rm
# `weights.zip` unpacks the same files into ./weights/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/weights}"
REPO_ID="${RAP_HF_REPO:-YuePanEdward/RAP}"
# Which checkpoints to fetch (the L model + the mini-SpinNet feature net by
# default; add rap_model_10.ckpt for the M model).
FILES="${RAP_FILES:-rap_model_12.ckpt mini_spinnet_t.pth}"
mkdir -p "$DEST"

# Prefer the official huggingface_hub downloader (handles auth/caching/resume).
if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  for rel in $FILES; do
    echo "downloading ${REPO_ID}:${rel}"
    python3 - "$REPO_ID" "$rel" "$DEST" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo_id, rel, dest = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo_id, filename=rel, local_dir=dest)
print("  ->", p)
PY
  done
  echo "done: weights under $DEST/"
  exit 0
fi

# Fallback A: direct HuggingFace resolve URLs (no huggingface_hub installed).
BASE="https://huggingface.co/${REPO_ID}/resolve/main"
fetch() {  # fetch <url> <out>
  echo "downloading $1 -> $2"
  if command -v curl >/dev/null 2>&1; then curl -L --fail -o "$2" "$1"
  else wget -O "$2" "$1"; fi
}
if [ "${RAP_USE_IPB:-0}" != "1" ]; then
  for rel in $FILES; do fetch "${BASE}/${rel}" "$DEST/$rel"; done
  echo "done: weights under $DEST/"
  exit 0
fi

# Fallback B: ipb.uni-bonn.de weights.zip (set RAP_USE_IPB=1).
ZIP="$DEST/weights.zip"
fetch "https://www.ipb.uni-bonn.de/html/projects/rap/weights.zip" "$ZIP"
unzip -o "$ZIP" -d "$DEST"
rm -f "$ZIP"
echo "done: weights under $DEST/"
