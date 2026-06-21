#!/bin/bash
#SBATCH --job-name=p2s_ok_sel10
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/points2surf_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

python -u scripts/run_points2surf_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --input-points 200000 \
  --sample-points 200000 \
  --query-grid-resolution 64 \
  --epsilon 3 \
  --certainty-threshold 13 \
  --sigma 5 \
  --batch-size 501 \
  --workers 2
