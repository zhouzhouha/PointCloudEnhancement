#!/bin/bash
#SBATCH --job-name=gradpu_post_smoke
#SBATCH --account=project_465003117
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gradpu_post_smoke_%j.out
#SBATCH --error=logs/gradpu_post_smoke_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G

PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env
PCQM_BIN=${REPO_ROOT}/third_party/metrics/PCQM/build_lumi_rocm/PCQM
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
  "${SIF}" "${VENV}/bin/python" - <<'PY'
import sys
from pathlib import Path
import numpy as np
import scipy
import skimage
import torch
import lpips

repo = Path('/project/project_465003117/xuemei/PointCloudEnhancement')
sys.path.insert(0, str(repo / 'third_party/UVG-CWI-Metric'))
from metrics import estimate_normals_knn

print('torch', torch.__version__, 'hip', torch.version.hip)
print('device', torch.cuda.get_device_name(0))
print('numpy', np.__version__, 'scipy', scipy.__version__, 'skimage', skimage.__version__)
model = lpips.LPIPS(net='alex').to('cuda').eval()
x = torch.zeros(1, 3, 64, 64, device='cuda')
y = torch.ones(1, 3, 64, 64, device='cuda')
with torch.no_grad():
    value = float(model(x, y).item())
assert np.isfinite(value)
points = np.random.default_rng(21).normal(size=(64, 3)).astype(np.float32)
normals = estimate_normals_knn(points, k=20, batch_size=64)
assert normals.shape == (64, 3) and np.isfinite(normals).all()
print('lpips', value, 'normal_shape', normals.shape, 'normal_finite', True)
PY

[[ -x "${PCQM_BIN}" ]]
singularity exec \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  "${SIF}" /bin/bash -lc "set -o pipefail; linked=\$(ldd '${PCQM_BIN}'); echo \"\${linked}\"; if grep -q 'not found' <<<\"\${linked}\"; then exit 1; fi"
echo "PCQM_LINK_CHECK=PASS"
