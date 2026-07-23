#!/bin/bash
#SBATCH --job-name=pathnet_missing731
#SBATCH --account=project_465003117
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-17%4
#SBATCH --output=logs/pathnet_missing731_%A_%a.out
#SBATCH --error=logs/pathnet_missing731_%A_%a.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G

PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
FLASH_ROOT=/flash/project_465003117/PointCloudEnhancement
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env
METHOD_PREFIX=pathnet_chunked_full_completion_20260721

SEQUENCES=(
  FitFluencer PinkNoir
  TicTacToe TicTacToe TicTacToe TicTacToe
  TrumanShow TrumanShow TrumanShow TrumanShow
  VictoryHeart VictoryHeart VictoryHeart VictoryHeart
  VirtualLife VirtualLife VirtualLife VirtualLife
)
BATCHES=(4 4 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3)
SEQUENCE=${SEQUENCES[$SLURM_ARRAY_TASK_ID]}
BATCH_INDEX=${BATCHES[$SLURM_ARRAY_TASK_ID]}
METHOD_NAME=$(printf '%s_batch_%02d' "${METHOD_PREFIX}" "${BATCH_INDEX}")
TARGET=${FLASH_ROOT}/results/method_outputs/${METHOD_NAME}/${SEQUENCE}/15fps
if [[ -e "${TARGET}" ]]; then
  echo "Refusing to overwrite immutable output directory: ${TARGET}" >&2
  exit 2
fi

export MIOPEN_USER_DB_PATH=/scratch/project_465003117/PointCloudEnhancement/miopen/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"
cd "${REPO_ROOT}"

srun singularity exec \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "/flash/project_465003117:/flash/project_465003117" \
  --bind "/scratch/project_465003117:/scratch/project_465003117" \
  --env "MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH}" \
  --env "MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR}" \
  "${SIF}" "${VENV}/bin/python" -u scripts/run_pathnet_full_dataset_batch.py \
    --sequence "${SEQUENCE}" --batch-index "${BATCH_INDEX}" --batch-size 50 \
    --dataset-root "${FLASH_ROOT}/data/UVG_CWI_DQPC/UVG-CWI-DQPC" \
    --results-root "${FLASH_ROOT}/results" \
    --method-prefix "${METHOD_PREFIX}"
