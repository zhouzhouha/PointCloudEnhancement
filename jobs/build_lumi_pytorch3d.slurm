#!/bin/bash
#SBATCH --job-name=build_pytorch3d
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_lumi_pytorch3d_%j.out
#SBATCH --error=logs/build_lumi_pytorch3d_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
mkdir -p logs

# PyTorch3D v0.7.8 officially supports PyTorch 2.1 through 2.4. Keep the
# working PyTorch 2.4.1/CUDA 12.1 stack fixed and compile only PyTorch3D.
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

"${PCE_PYTHON[@]}" -m pip install iopath
"${PCE_PYTHON[@]}" -m pip install --no-deps --no-build-isolation \
  "https://github.com/facebookresearch/pytorch3d/archive/refs/tags/V0.7.8.tar.gz"

"${PCE_PYTHON[@]}" -c \
  "import torch, pytorch3d; from pytorch3d.ops import knn_points; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('pytorch3d', pytorch3d.__version__); print('knn_points', knn_points.__name__)"
