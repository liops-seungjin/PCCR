#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <experiment_id> [cuda_visible_devices] [additional test.py args...]"
  echo "Example: $0 threedmatch 0 --pose_estimator ransac --verbose"
  exit 1
fi

EXPERIMENT_ID="$1"
shift || true

CUDA_VISIBLE_DEVICES_VALUE="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  CUDA_VISIBLE_DEVICES_VALUE="$1"
  shift || true
fi
EXTRA_ARGS=("$@")

DATASETS=(
  "3DMatch"
  "3DLoMatch"
  "Scannetpp_iphone"
  "Scannetpp_faro"
  "TIERS"
  "KITTI"
  "WOD"
  "MIT"
  "KAIST"
  "ETH"
  "Oxford"
)

DATASET_ARGS="${DATASETS[@]}"

echo "============================================================"
echo "Starting evaluation across all standard benchmark datasets"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0}" python test.py \
  --dataset $DATASET_ARGS \
  --experiment_id "$EXPERIMENT_ID" \
  --verbose \
  "${EXTRA_ARGS[@]}"
