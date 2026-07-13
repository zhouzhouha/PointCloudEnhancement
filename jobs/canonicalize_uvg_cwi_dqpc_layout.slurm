#!/bin/bash
#SBATCH --job-name=uvg_layout
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/canonicalize_uvg_cwi_dqpc_layout_%j.out
#SBATCH --error=logs/canonicalize_uvg_cwi_dqpc_layout_%j.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

DATA_ROOT=/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC
SEQUENCES=(
  BlueSpeech
  BlueVolley
  BouncingBlue
  FitFluencer
  GoodVision
  Mannequin
  OrangeKettlebell
  PinkNoir
  TicTacToe
  TrumanShow
  VictoryHeart
  VirtualLife
)

for seq in "${SEQUENCES[@]}"; do
  seq_root="${DATA_ROOT}/${seq}"
  he_src="${seq_root}/high-end_capture_system/HE/15fps"
  cg_src="${seq_root}/consumer-grade_capture_system/CGv2/15fps"
  he_link="${seq_root}/he/15fps"
  cg_link="${seq_root}/cg/15fps"

  if [ ! -d "$he_src" ]; then
    echo "[error] missing HE source: $he_src" >&2
    exit 1
  fi
  if [ ! -d "$cg_src" ]; then
    echo "[error] missing CGv2 source: $cg_src" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$he_link")" "$(dirname "$cg_link")"
  ln -sfn "$he_src" "$he_link"
  ln -sfn "$cg_src" "$cg_link"

  he_count=$(find -L "$he_link" -maxdepth 1 -name '*.ply' -type f | wc -l)
  cg_count=$(find -L "$cg_link" -maxdepth 1 -name '*.ply' -type f | wc -l)
  echo "${seq}: he=${he_count} cg=${cg_count} he_link=${he_link} cg_link=${cg_link}"
done

python -u scripts/build_uvg_dataset_manifest.py \
  --dataset-root "$DATA_ROOT" \
  --out results/full_dataset/uvg_cwi_dqpc_he15_cgv2_15_manifest.csv
