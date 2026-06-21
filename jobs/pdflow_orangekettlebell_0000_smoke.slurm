#!/bin/bash
#SBATCH --job-name=pdflow_ok_0000
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/pdflow_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0

srun python -u scripts/run_pdflow_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pdflow \
  --patch-size 1024 \
  --niters 1 \
  --cluster-size 10000
