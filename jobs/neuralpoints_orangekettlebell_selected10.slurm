#!/bin/bash
#SBATCH --job-name=neuralpoints_ok10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/neuralpoints_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export PYTHONUNBUFFERED=1

srun python -u scripts/run_neuralpoints_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --method-name neuralpoints_16x_2048_selected10 \
  --input-points 2048 \
  --device cuda
