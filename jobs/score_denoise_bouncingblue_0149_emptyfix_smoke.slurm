#!/bin/bash
#SBATCH --job-name=score_emptyfix
#SBATCH --account=mcsei7083
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs

module purge
module load 2023
module load CUDA/12.1.1

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/torch_extensions
export TMPDIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/tmp
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

srun python -u scripts/run_full_dataset_method.py \
  --method score_denoise \
  --method-name score_denoise_emptyfix_smoke \
  --sequence BouncingBlue \
  --frames 0149 \
  --dataset-root /gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results
