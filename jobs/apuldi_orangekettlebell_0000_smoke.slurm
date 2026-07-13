#!/bin/bash
#SBATCH --job-name=apuldi_smoke
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/apuldi_orangekettlebell_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export PCE_FORCE_APULDI_FALLBACKS=1
export TORCH_EXTENSIONS_DIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/torch_extensions
export TMPDIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/tmp
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"

srun python -u scripts/run_apuldi_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name apuldi_local_pu1k_4x_2048 \
  --input-points 2048 \
  --device cuda \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results
