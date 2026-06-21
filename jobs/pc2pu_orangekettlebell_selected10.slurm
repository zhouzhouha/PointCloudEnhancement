#!/bin/bash
#SBATCH --job-name=pc2pu_ok10
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pc2pu_ok_selected10_%j.out
#SBATCH --error=logs/pc2pu_ok_selected10_%j.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

/home/xzhou/.conda/envs/torch_env/bin/python scripts/run_pc2pu_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --method-name pc2pu_4x_chunks256_selected10 \
  --chunk-size 256 \
  --batch-chunks 16 \
  --device cuda \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results

mkdir -p results/method_outputs results/uvg_cwi_dqpc/OrangeKettlebell
ln -sfn /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/method_outputs/pc2pu_4x_chunks256_selected10 \
  results/method_outputs/pc2pu_4x_chunks256_selected10
ln -sfn /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/uvg_cwi_dqpc/OrangeKettlebell/pc2pu_4x_chunks256_selected10 \
  results/uvg_cwi_dqpc/OrangeKettlebell/pc2pu_4x_chunks256_selected10
