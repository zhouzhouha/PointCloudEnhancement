#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <archive-path>"
  echo "Example: $0 ~/lumi_missing_method_repos.tar.gz"
  exit 2
fi

ARCHIVE="$1"
PROJECT_DIR="$(pwd)"

REQUIRED_PATHS=(
  "third_party/enhancement/Grad-PU/pretrained_model/pu1k/ckpt/ckpt-epoch-60.pth"
  "third_party/enhancement/score-denoise/pretrained/ckpt.pt"
  "third_party/enhancement/P2P-Bridge/pretrained/PVDS_PUNet/latest.pth"
  "third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json"
  "third_party/enhancement/GQE-Net/pths/final_2023/GQE-Net/2023-07-25"
)

if [ ! -f "$ARCHIVE" ]; then
  echo "Archive not found: $ARCHIVE" >&2
  exit 1
fi

echo "[unpack] project directory: $PROJECT_DIR"
tar -xzf "$ARCHIVE"

echo "[verify] required paths"
for path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -e "$path" ]; then
    echo "MISSING $path" >&2
    exit 1
  fi
  echo "FOUND $path"
done

echo "Unpack and verification complete."
