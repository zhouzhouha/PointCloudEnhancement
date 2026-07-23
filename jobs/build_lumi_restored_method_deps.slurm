#!/bin/bash
#SBATCH --job-name=build_restored_deps
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_lumi_restored_method_deps_%j.out
#SBATCH --error=logs/build_lumi_restored_method_deps_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
mkdir -p logs

export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

# Pure-Python metric dependency used by GQE-Net.
"${PCE_PYTHON[@]}" -m pip install sewar

# PD-LTS imports its bundled CUDA EMD module during inference even though the
# benchmark wrapper does not use the EMD metric directly.
"${PCE_PYTHON[@]}" -m pip install --no-deps --no-build-isolation \
  "${PCE_REPO_ROOT}/third_party/enhancement/PD-LTS/metric/emd"

"${PCE_PYTHON[@]}" -c \
  "import sewar, emd; print('sewar', sewar.__file__); print('emd', emd.__file__)"
