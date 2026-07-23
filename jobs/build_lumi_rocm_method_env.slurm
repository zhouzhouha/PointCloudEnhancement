#!/bin/bash
#SBATCH --job-name=pce_rocm_env
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_lumi_rocm_env_%j.out
#SBATCH --error=logs/build_lumi_rocm_env_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/C
PROJECT_ROOT=/project/project_465003117
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV="${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env"
BASE=/opt/miniconda3/envs/pytorch/bin/python
test -r "${SIF}"
if [[ ! -f "${VENV}/pyvenv.cfg" ]]; then
  singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" "${BASE}" -m venv --system-site-packages "${VENV}"
fi
singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" "${VENV}/bin/python" -m pip install \
  scikit-learn h5py trimesh omegaconf loguru easydict sewar ema-pytorch shortuuid multimethod point-cloud-utils plyfile
singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" "${VENV}/bin/python" - <<'PY'
import torch, sklearn, h5py, trimesh, omegaconf, loguru, easydict
print("torch", torch.__version__, "hip", torch.version.hip)
print("ROCm method environment ready")
PY
