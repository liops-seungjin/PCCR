#!/usr/bin/env bash
# Fetch the BUFFER-X pretrained checkpoints into ./weights/ (next to the worker).
#
# The weights are a RUNTIME concern only; they are never required to build
# CloudCropper and are .gitignored.
#
# Recon (2026-06-16, docs/design/_bufferx-upstream-notes.md): checkpoints are
# hosted on HuggingFace `Hyungtae-Lim/BUFFER-X` under snapshot/{source}/{stage}/
# best.pth, organized by training SOURCE (each used zero-shot). The 3DMatch-
# trained model is the indoor/general zero-shot generalist; KITTI is outdoor.
# Files are tiny (~3.67 MB each, ~7.3 MB per model):
#   snapshot/threedmatch/Desc/best.pth   snapshot/threedmatch/Pose/best.pth
#   snapshot/kitti/Desc/best.pth         snapshot/kitti/Pose/best.pth
# Upstream's own fetcher: python scripts/download_pretrained_models.py \
#     --source hf --repo-id Hyungtae-Lim/BUFFER-X
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/weights}"
REPO_ID="${BUFFERX_HF_REPO:-Hyungtae-Lim/BUFFER-X}"
# Which source model(s) to fetch: threedmatch (indoor/general) and/or kitti.
SOURCES="${BUFFERX_SOURCES:-threedmatch}"
mkdir -p "$DEST"

# Prefer the official huggingface_hub downloader (handles auth/caching/resume).
if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  for src in $SOURCES; do
    for stage in Desc Pose; do
      rel="snapshot/${src}/${stage}/best.pth"
      echo "downloading ${REPO_ID}:${rel}"
      python3 - "$REPO_ID" "$rel" "$DEST" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo_id, rel, dest = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo_id, filename=rel, local_dir=dest)
print("  ->", p)
PY
    done
  done
  echo "done: weights under $DEST/snapshot/"
  exit 0
fi

# Fallback: direct HTTPS resolve URLs (no huggingface_hub installed).
BASE="https://huggingface.co/${REPO_ID}/resolve/main"
for src in $SOURCES; do
  for stage in Desc Pose; do
    rel="snapshot/${src}/${stage}/best.pth"
    out="$DEST/$rel"
    mkdir -p "$(dirname "$out")"
    echo "downloading ${BASE}/${rel} -> $out"
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail -o "$out" "${BASE}/${rel}"
    else
      wget -O "$out" "${BASE}/${rel}"
    fi
  done
done
echo "done: weights under $DEST/snapshot/"
