#!/bin/bash
#SBATCH --job-name=pathnet_normals
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-31%5
#SBATCH --output=logs/pathnet_normals_%A_%a.out
#SBATCH --error=logs/pathnet_normals_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"

SEQUENCES=(BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin OrangeKettlebell PinkNoir)
SEQUENCE_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % 4))

srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_missing_normal_metrics.py \
  --sequence "${SEQUENCES[$SEQUENCE_INDEX]}" \
  --batch-index "${BATCH_INDEX}" \
  --batch-size 50 \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --results-root "${PCE_RESULTS_ROOT}" \
  --normal-k 20
