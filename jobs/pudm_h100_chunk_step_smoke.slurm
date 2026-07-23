#!/bin/bash
#SBATCH --job-name=pudm_h100_sweep
#SBATCH --account=mcsei7083
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/pudm_h100_chunk_step_smoke_%A_%a.out
#SBATCH --error=logs/pudm_h100_chunk_step_smoke_%A_%a.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs

module purge
module load 2023
module load CUDA/12.1.1

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PCE_FORCE_PUDM_FALLBACKS=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/tmp
mkdir -p "$TMPDIR"

CHUNK_SIZES=(4096 8192 4096)
STEPS=(30 30 15)
METHOD_NAMES=(
  pudm_pu1k_4x_chunk4096_step30_smoke
  pudm_pu1k_4x_chunk8192_step30_smoke
  pudm_pu1k_4x_chunk4096_step15_smoke
)

CHUNK_SIZE="${CHUNK_SIZES[$SLURM_ARRAY_TASK_ID]}"
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
METHOD_NAME="${METHOD_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "[pudm-smoke] method=${METHOD_NAME} chunk=${CHUNK_SIZE} step=${STEP}"

srun python -u scripts/run_pudm_selected_frames.py \
  --sequence BouncingBlue \
  --frames 0149 \
  --dataset-root /gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC \
  --method-name "$METHOD_NAME" \
  --checkpoint third_party/enhancement/PUDM/pointnet2/pkls/pu1k.pkl \
  --config third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json \
  --chunk-size "$CHUNK_SIZE" \
  --up-rate 4 \
  --step "$STEP" \
  --gamma 0.5
