#!/bin/bash
#SBATCH --job-name=pathnet_full_b50
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/pathnet_full_batched_%A_%a.out
#SBATCH --error=logs/pathnet_full_batched_%A_%a.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"
mkdir -p logs

SEQUENCES=(
  BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin
  OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife
)

# Four 50-frame batches cover the longest sequence. Empty tail batches exit cleanly.
GLOBAL_TASK_ID=$((${PCE_TASK_OFFSET:-0} + SLURM_ARRAY_TASK_ID))
SEQUENCE_INDEX=$((GLOBAL_TASK_ID / 4))
BATCH_INDEX=$((GLOBAL_TASK_ID % 4))
SEQUENCE="${SEQUENCES[$SEQUENCE_INDEX]}"

srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_full_dataset_batch.py \
  --sequence "${SEQUENCE}" \
  --batch-index "${BATCH_INDEX}" \
  --batch-size 50 \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --results-root "${PCE_RESULTS_ROOT}"
