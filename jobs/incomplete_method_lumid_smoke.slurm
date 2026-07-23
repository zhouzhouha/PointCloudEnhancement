#!/bin/bash
#SBATCH --job-name=pce_method_smoke
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lumid_smoke_%x_%j.out
#SBATCH --error=logs/lumid_smoke_%x_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"

: "${METHOD:?submit with --export=ALL,METHOD=<method>}"
SEQUENCE="${SEQUENCE:-BlueSpeech}"
FRAME="${FRAME:-0000}"

srun "${PCE_PYTHON[@]}" -u scripts/run_full_dataset_method.py \
  --method "${METHOD}" \
  --sequence "${SEQUENCE}" \
  --frames "${FRAME}" \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --results-root "${PCE_RESULTS_ROOT}"
