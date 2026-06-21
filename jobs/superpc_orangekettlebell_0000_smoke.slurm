#!/bin/bash
#SBATCH --job-name=superpc_smoke
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --output=logs/superpc_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/torch_extensions_superpc"
export CUDA_HOME="${CUDA_HOME:-/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1}"
export PYTHONPATH="/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/SuperPC:${PYTHONPATH:-}"

srun python -u scripts/run_superpc_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name superpc_tartanair_novision_4x \
  --checkpoint third_party/enhancement/SuperPC/pretrained/tartanair_com.pth \
  --superpc-dataset tartanair \
  --num-points 11520 \
  --target-num-points 46080 \
  --sampling-steps 25
