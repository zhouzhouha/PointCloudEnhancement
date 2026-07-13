#!/bin/bash
#SBATCH --job-name=pce_full_sum
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_dataset_summarize_%j.out
#SBATCH --error=logs/full_dataset_summarize_%j.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

python -u scripts/build_uvg_dataset_manifest.py \
  --dataset-root /gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC \
  --out results/full_dataset/uvg_cwi_dqpc_he15_cgv2_15_manifest.csv

mapfile -t GEOM < <(find /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results/uvg_cwi_dqpc -path '*/per_frame_metrics.csv' -type f | sort)
mapfile -t TEX < <(find results/texture_perceptual_metrics -path '*/full_dataset/per_frame_texture_perceptual_metrics.csv' -type f | sort)

python -u scripts/summarize_full_dataset_results.py \
  --geometry "${GEOM[@]}" \
  --texture "${TEX[@]}" \
  --out-dir results/full_dataset/summary
