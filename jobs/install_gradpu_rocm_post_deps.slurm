#!/bin/bash
#SBATCH --job-name=gradpu_post_deps
#SBATCH --account=project_465003117
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gradpu_post_deps_%j.out
#SBATCH --error=logs/gradpu_post_deps_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G
PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env

singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" \
  "${VENV}/bin/python" -m pip install --upgrade \
  'numpy==1.26.4' 'scikit-image==0.25.2' 'lpips==0.1.4' \
  'plyfile==1.0.3' 'tifffile==2025.6.11'

singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" \
  "${VENV}/bin/python" -m pip check

cd "${REPO_ROOT}"
singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" \
  "${VENV}/bin/python" - <<'PY'
import lpips
import numpy
import scipy
import skimage
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('skimage', skimage.__version__)
print('lpips', lpips.__file__)
PY
