#!/bin/bash
#SBATCH --job-name=pudense_me
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/pudense_build_minkowski_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

export CONDA_PREFIX=/home/xzhou/.conda/envs/pudense
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export PIP_CACHE_DIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/pip_cache
export TMPDIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/tmp

mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

"${CONDA_PREFIX}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

"${CONDA_PREFIX}/bin/pip" install -v --no-deps --no-cache-dir \
  "MinkowskiEngine==0.5.4" \
  --install-option="--force_cuda" \
  --install-option="--blas=openblas"

"${CONDA_PREFIX}/bin/python" - <<'PY'
import MinkowskiEngine as ME
print("MinkowskiEngine", getattr(ME, "__version__", "unknown"))
PY
