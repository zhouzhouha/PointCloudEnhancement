#!/bin/bash
#SBATCH --job-name=pc2pu_ok0000
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pc2pu_ok0000_%j.out
#SBATCH --error=logs/pc2pu_ok0000_%j.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

/home/xzhou/.conda/envs/torch_env/bin/python scripts/run_pc2pu_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pc2pu_4x_chunks256 \
  --chunk-size 256 \
  --batch-chunks 16 \
  --device cuda
