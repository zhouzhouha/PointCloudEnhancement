#!/bin/bash
#SBATCH --job-name=itpfn_ok_0000
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/iterativepfn_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

srun python -u scripts/run_iterativepfn_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name iterativepfn \
  --patch-size 1000 \
  --niters 1 \
  --seed-k 6 \
  --seed-k-alpha 10
