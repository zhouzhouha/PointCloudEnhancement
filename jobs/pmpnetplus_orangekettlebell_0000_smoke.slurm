#!/bin/bash
#SBATCH --job-name=pmpplus_ok_0000
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pmpplus_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_pmpnet_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pmpnetplus_pcn \
  --checkpoint third_party/enhancement/PMP-Net/pretrained/pcn/ckpt-best-pmpplus.pth \
  --model-variant pmpnetplus \
  --n-input 2048 \
  --n-repeats 8 \
  --seed 2026
