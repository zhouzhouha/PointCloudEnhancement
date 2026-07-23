#!/bin/bash
#SBATCH --job-name=pce_cuda_env
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_lumi_cuda_environment_%j.out
#SBATCH --error=logs/build_lumi_cuda_environment_%j.err

set -euo pipefail

REPO_ROOT=/project/project_465003117/xuemei/PointCloudEnhancement
CONTAINER_DIR=/project/project_465003117/containers
CONTAINER="${CONTAINER_DIR}/pce-pytorch-2.4.1-cu121-devel.sif"
VENV=/project/project_465003117/envs/PointCloudEnhancement/lumi_cuda_torch_env
IMAGE=docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

module --force purge
module load LUMI/25.03 partition/C

mkdir -p "${CONTAINER_DIR}" "$(dirname "${VENV}")" "${REPO_ROOT}/logs"

if [[ ! -s "${CONTAINER}" ]]; then
  tmp_container="${CONTAINER}.part"
  rm -f "${tmp_container}"
  singularity pull "${tmp_container}" "${IMAGE}"
  mv "${tmp_container}" "${CONTAINER}"
fi

if [[ ! -f "${VENV}/pyvenv.cfg" ]]; then
  singularity exec \
    --bind /project/project_465003117:/project/project_465003117 \
    "${CONTAINER}" \
    /opt/conda/bin/python -m venv --system-site-packages "${VENV}"
fi

PYTHON=(
  singularity exec --cleanenv
  --bind /project/project_465003117:/project/project_465003117
  --env "VIRTUAL_ENV=${VENV}"
  --env "PATH=${VENV}/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
  "${CONTAINER}"
  "${VENV}/bin/python"
)

"${PYTHON[@]}" -m pip install --upgrade pip setuptools wheel
# Open3D 0.19 wheels require X11 runtime libraries absent from this CUDA base
# image. Keep it out of the core environment; methods that require Open3D need
# a separately recorded runtime layer. PathNet and UVG-CWI/Metric do not use it.
"${PYTHON[@]}" -m pip uninstall --yes open3d open3d-cpu || true
"${PYTHON[@]}" -m pip install -r "${REPO_ROOT}/skills/methods/lumi_cuda_metric_requirements.txt"
"${PYTHON[@]}" -m pip freeze
"${PYTHON[@]}" - <<'PY'
import lpips
import numpy
import plyfile
import scipy
import skimage
import sklearn
import torch
import torchvision
import trimesh

print("python environment imports: OK")
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available_on_cpu_job", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("skimage", skimage.__version__)
print("sklearn", sklearn.__version__)
print("torchvision", torchvision.__version__)
print("trimesh", trimesh.__version__)
PY
