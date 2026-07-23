#!/bin/bash
#SBATCH --job-name=pce_sum_c12
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_dataset_summary_completed12_%j.out
#SBATCH --error=logs/full_dataset_summary_completed12_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
mkdir -p logs

METHODS=(
  apuldi_local_pu1k_4x_2048_full
  neuralpoints_16x_2048_full
  pucrn_pu1k_4x_full
  pufm_pugan_4x_full
  puflow_discrete_full
  crcir_aftercomp_4x_full
  upsample_clean_ounet_full
  spu_pointnet_4x_full
)

GEOMETRY=()
for method in "${METHODS[@]}"; do
  while IFS= read -r path; do
    GEOMETRY+=("$path")
  done < <(find "${PCE_RESULTS_ROOT}/uvg_cwi_dqpc" \
    -mindepth 3 -maxdepth 3 -path "*/${method}/per_frame_metrics.csv" -print | sort)
done

TEXTURE=()
while IFS= read -r path; do
  TEXTURE+=("$path")
done < <(find results/texture_perceptual_metrics \
  -path "*/full_dataset_completed12_interim/per_frame_texture_perceptual_metrics.csv" -print | sort)

srun "${PCE_PYTHON[@]}" -u scripts/summarize_full_dataset_results.py \
  --geometry "${GEOMETRY[@]}" \
  --texture "${TEXTURE[@]}" \
  --out-dir results/full_dataset/summary_completed12_interim
