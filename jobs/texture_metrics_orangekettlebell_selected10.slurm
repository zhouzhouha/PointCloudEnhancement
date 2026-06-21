#!/bin/bash
#SBATCH --job-name=texmet_ok_sel10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/texture_metrics_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_texture_perceptual_metrics.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --methods gqenet_selected10 mag_selected10 score_denoise_selected10 pointcleannet_selected10 upsample_clean_ounet_selected10 \
  --run-name selected10 \
  --image-size 512
