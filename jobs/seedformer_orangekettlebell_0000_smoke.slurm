#!/bin/bash
#SBATCH --job-name=seedformer_ok_0000
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/seedformer_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_seedformer_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name seedformer_pcn \
  --n-input 2048 \
  --seed 2026
