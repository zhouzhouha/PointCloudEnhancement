#!/bin/bash
#SBATCH --job-name=build_pytorch3d_cuda
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_lumi_pytorch3d_cuda_%j.out
#SBATCH --error=logs/build_lumi_pytorch3d_cuda_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"

export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

"${PCE_PYTHON[@]}" -m pip install --upgrade iopath
"${PCE_PYTHON[@]}" -m pip install --force-reinstall --no-deps --no-build-isolation \
  "https://github.com/facebookresearch/pytorch3d/archive/refs/tags/V0.7.8.tar.gz"

srun "${PCE_PYTHON[@]}" - <<'PY'
import torch
from pytorch3d.ops import knn_points, sample_farthest_points

assert torch.cuda.is_available()
x = torch.randn(1, 64, 3, device="cuda")
knn = knn_points(x, x, K=8)
sampled, indices = sample_farthest_points(x, K=16)
torch.cuda.synchronize()
assert torch.isfinite(knn.dists).all()
assert sampled.shape == (1, 16, 3)
print("device", torch.cuda.get_device_name(0))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("PyTorch3D CUDA KNN/FPS: OK", indices.shape)
PY
