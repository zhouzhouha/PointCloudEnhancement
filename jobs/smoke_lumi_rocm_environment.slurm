#!/bin/bash
#SBATCH --job-name=pce_rocm_smoke
#SBATCH --account=project_465003117
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_lumi_rocm_%j.out
#SBATCH --error=logs/smoke_lumi_rocm_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G
PROJECT_ROOT=/project/project_465003117
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV="${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env"
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
srun singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" "${VENV}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("available", torch.cuda.is_available())
print("count", torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("device", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print("matmul", float((x @ x).mean()))
PY
