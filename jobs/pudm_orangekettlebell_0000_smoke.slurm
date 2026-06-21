#!/bin/bash
#SBATCH --job-name=pudm_ok0
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=10:00:00
#SBATCH --output=logs/pudm_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export PCE_FORCE_PUDM_FALLBACKS=1

srun python -u scripts/run_pudm_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pudm_pu1k_4x \
  --checkpoint third_party/enhancement/PUDM/pointnet2/pkls/pu1k.pkl \
  --config third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json \
  --chunk-size 2048 \
  --up-rate 4 \
  --step 30 \
  --gamma 0.5
