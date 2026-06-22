#!/bin/bash
#SBATCH --job-name=pathnet_ok10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pathnet_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_pathnet_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results \
  --method-name pathnet_chunked_selected10 \
  --chunk-size 50000 \
  --batch-size 512 \
  --knn 128 \
  --iterations 2
