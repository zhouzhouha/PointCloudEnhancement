#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <lumi-destination-project-dir>"
  echo "Example: $0 user@lumi.csc.fi:/scratch/project_x/PointCloudEnhancement"
  exit 2
fi

DEST="${1%/}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_BIN=(ssh -i "$HOME/.ssh/id_ed25519_lumi" -o IdentitiesOnly=yes)
RSYNC_RSH="ssh -i $HOME/.ssh/id_ed25519_lumi -o IdentitiesOnly=yes"

METHOD_DIRS=(
  "third_party/enhancement/Grad-PU"
  "third_party/enhancement/score-denoise"
  "third_party/enhancement/P2P-Bridge"
  "third_party/enhancement/PUDM"
  "third_party/enhancement/GQE-Net"
)

REQUIRED_PATHS=(
  "third_party/enhancement/Grad-PU/pretrained_model/pu1k/ckpt/ckpt-epoch-60.pth"
  "third_party/enhancement/score-denoise/pretrained/ckpt.pt"
  "third_party/enhancement/P2P-Bridge/pretrained/PVDS_PUNet/latest.pth"
  "third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json"
  "third_party/enhancement/GQE-Net/pths/final_2023/GQE-Net/2023-07-25"
)

cd "$ROOT"

echo "[local] checking required source paths"
for path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -e "$path" ]; then
    echo "Missing local source path: $path" >&2
    exit 1
  fi
  echo "  ok: $path"
done

echo "[remote] creating destination directories"
"${SSH_BIN[@]}" "${DEST%%:*}" "mkdir -p '${DEST#*:}/third_party/enhancement'"

echo "[rsync] transferring complete method directories"
for dir in "${METHOD_DIRS[@]}"; do
  echo "  $dir"
  rsync -aH --info=progress2 -e "$RSYNC_RSH" "$dir/" "$DEST/$dir/"
done

echo "[remote] verifying required paths"
remote_check='set -euo pipefail; cd "$1"; shift; for path in "$@"; do if [ -e "$path" ]; then echo "FOUND $path"; else echo "MISSING $path"; exit 1; fi; done'
"${SSH_BIN[@]}" "${DEST%%:*}" bash -s -- "${DEST#*:}" "${REQUIRED_PATHS[@]}" <<< "$remote_check"

echo "Transfer complete."
