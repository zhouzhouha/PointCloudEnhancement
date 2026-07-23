#!/bin/bash
#SBATCH --job-name=gradpu_rocm_smoke
#SBATCH --account=project_465003117
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gradpu_rocm_smoke_%j.out
#SBATCH --error=logs/gradpu_rocm_smoke_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G

PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
FLASH_ROOT=/flash/project_465003117/PointCloudEnhancement
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env
METHOD=gradpu_rocm_smoke_20260723
TARGET=${FLASH_ROOT}/results/method_outputs/${METHOD}/BlueVolley/15fps

if [[ -e "${TARGET}" ]]; then
  echo "Refusing to overwrite immutable smoke output: ${TARGET}" >&2
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
  "${SIF}" "${VENV}/bin/python" -u scripts/run_gradpu_selected_frames.py \
    --sequence BlueVolley --frames 0100 \
    --dataset-root "${FLASH_ROOT}/data/UVG_CWI_DQPC/UVG-CWI-DQPC" \
    --results-root "${FLASH_ROOT}/results" \
    --method-name "${METHOD}" \
    --up-rate 4 --chunk-size 2048
