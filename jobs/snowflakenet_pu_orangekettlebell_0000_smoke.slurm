#!/bin/bash
#SBATCH --job-name=snowpu_ok_0000
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/snowflakenet_pu_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_snowflakenet_pu_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name snowflakenet_pu_4x \
  --checkpoint third_party/enhancement/SnowflakeNet/pretrained/pu/ckpt-pu.pth \
  --chunk-size 8192 \
  --patch-num-ratio 3 \
  --num-per-patch 256 \
  --up-ratio 4
