#!/bin/bash
#SBATCH --job-name=batchnorm_k20
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-251%5
#SBATCH --output=logs/batchnorm_k20_%A_%a.out
#SBATCH --error=logs/batchnorm_k20_%A_%a.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"

METHOD_PREFIX=${METHOD_PREFIX:?METHOD_PREFIX is required}
SEQUENCES=(
  BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin
  OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife
)
BATCHES_PER_SEQUENCE=21
SEQUENCE_INDEX=$((SLURM_ARRAY_TASK_ID / BATCHES_PER_SEQUENCE))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % BATCHES_PER_SEQUENCE))
SEQUENCE=${SEQUENCES[$SEQUENCE_INDEX]}
METHOD_NAME=$(printf '%s_batch_%02d' "${METHOD_PREFIX}" "${BATCH_INDEX}")
PRED_DIR="${PCE_RESULTS_ROOT}/method_outputs/${METHOD_NAME}/${SEQUENCE}/15fps"

if [[ ! -d "${PRED_DIR}" ]]; then
  echo "Missing immutable method output directory: ${PRED_DIR}" >&2
  exit 2
fi

# Every immutable method directory represents one at-most-ten-frame batch, so
# batch-index 0 intentionally selects every prediction contained in that method.
srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_missing_normal_metrics.py \
  --sequence "${SEQUENCE}" --batch-index 0 --batch-size 10 \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}" \
  --method "${METHOD_NAME}" \
  --normal-cache-root "${PCE_FLASH_ROOT}/data_cache/uvg_cwi_dqpc_normals" \
  --normal-k 20 --run-id "run_${SLURM_ARRAY_JOB_ID}_pca_knn20"
