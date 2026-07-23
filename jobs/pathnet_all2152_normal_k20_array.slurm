#!/bin/bash
#SBATCH --job-name=pathnet2152_norm
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-59%5
#SBATCH --output=logs/pathnet2152_norm_%A_%a.out
#SBATCH --error=logs/pathnet2152_norm_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
SEQUENCES=(BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife)
SEQUENCE_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_missing_normal_metrics.py \
  --sequence "${SEQUENCES[$SEQUENCE_INDEX]}" \
  --batch-index "${BATCH_INDEX}" --batch-size 50 \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}" \
  --method pathnet_chunked_full_all2152_20260721 \
  --normal-cache-root "${PCE_FLASH_ROOT}/data_cache/uvg_cwi_dqpc_normals" \
  --normal-k 20 --run-id "run_${SLURM_ARRAY_JOB_ID}_pca_knn20"
