#!/bin/bash
#SBATCH --job-name=crcir_ok10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/crcir_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export PYTHONPATH="/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/CRCIR_for_PCGC/python_deps:/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/CRCIR_for_PCGC:${PYTHONPATH:-}"

srun python -u scripts/run_crcir_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --method-name crcir_aftercomp_4x_selected10 \
  --chunk-size 2048 \
  --encoder-downsample 2 \
  --decoder-multiplier 4 \
  --device cuda
