#!/bin/bash
#SBATCH --job-name=pufm_ok0
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pufm_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export PCE_FORCE_PUFM_FALLBACKS=1

srun python -u scripts/run_pufm_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pufm_pugan_4x \
  --model pufm \
  --chunk-size 2048 \
  --up-rate 4 \
  --num-points 256 \
  --steps 5
