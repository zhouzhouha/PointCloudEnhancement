#!/bin/bash
#SBATCH --job-name=texmet_ps10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/texture_metrics_ok_selected10_pudm_spupmd_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_texture_perceptual_metrics.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --methods pudm_pu1k_4x_selected10 spupmd_pu1k_4x_selected10 \
  --run-name selected10_pudm_spupmd \
  --image-size 512 \
  --tmp-dir "${TMPDIR:-/tmp}"
