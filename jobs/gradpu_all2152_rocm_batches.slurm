#!/bin/bash
#SBATCH --job-name=gradpu2152_rocm
#SBATCH --account=project_465003117
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-251%2
#SBATCH --output=logs/gradpu2152_rocm_%A_%a.out
#SBATCH --error=logs/gradpu2152_rocm_%A_%a.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G

PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
FLASH_ROOT=/flash/project_465003117/PointCloudEnhancement
DATASET_ROOT=${FLASH_ROOT}/data/UVG_CWI_DQPC/UVG-CWI-DQPC
RESULTS_ROOT=${FLASH_ROOT}/results
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env

SEQUENCES=(
  BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin
  OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife
)
BATCH_SIZE=10
BATCHES_PER_SEQUENCE=21
METHOD_PREFIX=${METHOD_PREFIX:-gradpu_chunked_4x_full_20260723_rocm1}
SEQUENCE_INDEX=$((SLURM_ARRAY_TASK_ID / BATCHES_PER_SEQUENCE))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % BATCHES_PER_SEQUENCE))
SEQUENCE=${SEQUENCES[$SEQUENCE_INDEX]}
METHOD_NAME=$(printf '%s_batch_%02d' "${METHOD_PREFIX}" "${BATCH_INDEX}")

FRAME_TEXT=$(singularity exec \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "/flash/project_465003117:/flash/project_465003117" \
  --bind "/scratch/project_465003117:/scratch/project_465003117" \
  "${SIF}" "${VENV}/bin/python" - "${DATASET_ROOT}" "${SEQUENCE}" \
  "${BATCH_INDEX}" "${BATCH_SIZE}" <<'PY'
from pathlib import Path
import sys

root, sequence, batch_index, batch_size = sys.argv[1:]
batch_index, batch_size = int(batch_index), int(batch_size)
cg = {p.stem.rsplit('_', 1)[-1] for p in (Path(root) / sequence / 'cg' / '15fps').glob('*.ply')}
he = {p.stem.rsplit('_', 1)[-1] for p in (Path(root) / sequence / 'he' / '15fps').glob('*.ply')}
frames = sorted(cg & he)[batch_index * batch_size:(batch_index + 1) * batch_size]
print('\n'.join(frames))
PY
)
mapfile -t FRAMES <<<"${FRAME_TEXT}"

if [[ -z "${FRAME_TEXT}" || ${#FRAMES[@]} -eq 0 ]]; then
  echo "No paired frames for ${SEQUENCE} batch ${BATCH_INDEX}; nothing to run."
  exit 0
fi

TARGET=${RESULTS_ROOT}/method_outputs/${METHOD_NAME}/${SEQUENCE}/15fps
if [[ -e "${TARGET}" ]]; then
  echo "Refusing to overwrite immutable ROCm batch output: ${TARGET}" >&2
  exit 2
fi

export MIOPEN_USER_DB_PATH=/scratch/project_465003117/PointCloudEnhancement/miopen/${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export PCE_GRADPU_KNN_QUERY_CHUNK=2048
mkdir -p "${MIOPEN_USER_DB_PATH}"
cd "${REPO_ROOT}"

srun singularity exec \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "/flash/project_465003117:/flash/project_465003117" \
  --bind "/scratch/project_465003117:/scratch/project_465003117" \
  --env "MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH}" \
  --env "MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR}" \
  --env "PCE_GRADPU_KNN_QUERY_CHUNK=${PCE_GRADPU_KNN_QUERY_CHUNK}" \
  "${SIF}" "${VENV}/bin/python" -u scripts/run_full_dataset_method.py \
    --method gradpu --method-name "${METHOD_NAME}" \
    --sequence "${SEQUENCE}" --frames "${FRAMES[@]}" \
    --dataset-root "${DATASET_ROOT}" --results-root "${RESULTS_ROOT}"
