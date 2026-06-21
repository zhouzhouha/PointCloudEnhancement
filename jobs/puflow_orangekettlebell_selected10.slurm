#!/bin/bash
#SBATCH --job-name=puflow_ok10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/puflow_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_puflow_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --method-name puflow_discrete_selected10 \
  --chunk-size 8192 \
  --num-patch 256 \
  --up-ratio 4
