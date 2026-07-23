#!/bin/bash
#SBATCH --job-name=pce_full_sum
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_dataset_summarize_%j.out
#SBATCH --error=logs/full_dataset_summarize_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"

"${PCE_PYTHON[@]}" -u scripts/build_uvg_dataset_manifest.py \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --out results/full_dataset/uvg_cwi_dqpc_he15_cgv2_15_manifest.csv

mapfile -t GEOM < <(find "${PCE_RESULTS_ROOT}/uvg_cwi_dqpc" -path '*/per_frame_metrics.csv' -type f | sort)
mapfile -t TEX < <(find results/texture_perceptual_metrics -path '*/full_dataset/per_frame_texture_perceptual_metrics.csv' -type f | sort)

srun "${PCE_PYTHON[@]}" -u scripts/summarize_full_dataset_results.py \
  --geometry "${GEOM[@]}" \
  --texture "${TEX[@]}" \
  --out-dir results/full_dataset/summary
