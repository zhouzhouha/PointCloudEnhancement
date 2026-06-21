#!/bin/bash
#SBATCH --job-name=spupmd_ok10
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spupmd_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export PCE_FORCE_SPUPMD_FALLBACKS=1

srun python -u scripts/run_spupmd_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --method-name spupmd_pu1k_4x_selected10 \
  --checkpoint third_party/enhancement/SPU-PMD/pretrained/spupmd_pu1k_pretrained \
  --chunk-size 2048 \
  --num-point 256 \
  --up-ratio 4 \
  --patch-num-ratio 3
