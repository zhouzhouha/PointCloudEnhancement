#!/bin/bash
#SBATCH --job-name=pdlts_ok_0000
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pdlts_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

module purge
module load 2023
module load CUDA/12.1.1

export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/build/pdlts_torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

srun /home/xzhou/.conda/envs/torch_env/bin/python -u scripts/run_pdlts_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pdlts_light_fbm \
  --variant light \
  --patch-size 1024 \
  --niters 1 \
  --seed-k 5 \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results
