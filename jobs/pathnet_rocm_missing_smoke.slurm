#!/bin/bash
#SBATCH --job-name=pathnet_rocm_smoke
#SBATCH --account=project_465003117
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pathnet_rocm_smoke_%j.out
#SBATCH --error=logs/pathnet_rocm_smoke_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G

PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
FLASH_ROOT=/flash/project_465003117/PointCloudEnhancement
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env
export MIOPEN_USER_DB_PATH=/scratch/project_465003117/PointCloudEnhancement/miopen/${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"
cd "${REPO_ROOT}"

srun singularity exec \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "/flash/project_465003117:/flash/project_465003117" \
  --bind "/scratch/project_465003117:/scratch/project_465003117" \
  --env "MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH}" \
  --env "MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR}" \
  "${SIF}" "${VENV}/bin/python" -u scripts/run_pathnet_selected_frames.py \
    --sequence TicTacToe --frames 0000 \
    --dataset-root "${FLASH_ROOT}/data/UVG_CWI_DQPC/UVG-CWI-DQPC" \
    --results-root "${FLASH_ROOT}/results" \
    --method-name pathnet_rocm_smoke_20260721 \
    --chunk-size 50000 --batch-size 512 --knn 128 --iterations 2 --device cuda
