#!/bin/bash
#SBATCH --job-name=gradpu_h100_0000
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gradpu_h100_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_gradpu_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name gradpu_h100 \
  --up-rate 4
