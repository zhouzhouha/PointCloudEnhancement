#!/bin/bash
#SBATCH --job-name=pudm8192_full
#SBATCH --account=mcsei7083
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-8%2
#SBATCH --output=logs/pudm_h100_chunk8192_step30_full_%A_%a.out
#SBATCH --error=logs/pudm_h100_chunk8192_step30_full_%A_%a.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs

SEQUENCES=(
  BlueSpeech
  BlueVolley
  BouncingBlue
  OrangeKettlebell
  PinkNoir
  TicTacToe
  TrumanShow
  VictoryHeart
  VirtualLife
)

SEQUENCE="${SEQUENCES[$SLURM_ARRAY_TASK_ID]}"
METHOD_NAME=pudm_pu1k_4x_chunk8192_step30_full
PROJECT_RESULTS=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results

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

mkdir -p \
  "$PROJECT_RESULTS/method_outputs/$METHOD_NAME" \
  "$PROJECT_RESULTS/work/$METHOD_NAME" \
  "$PROJECT_RESULTS/uvg_cwi_dqpc/$SEQUENCE/$METHOD_NAME"
ln -sfn "$PROJECT_RESULTS/method_outputs/$METHOD_NAME" "results/method_outputs/$METHOD_NAME"
ln -sfn "$PROJECT_RESULTS/work/$METHOD_NAME" "results/work/$METHOD_NAME"
ln -sfn "$PROJECT_RESULTS/uvg_cwi_dqpc/$SEQUENCE/$METHOD_NAME" "results/uvg_cwi_dqpc/$SEQUENCE/$METHOD_NAME"

echo "[pudm-full] method=${METHOD_NAME} sequence=${SEQUENCE} chunk=8192 step=30 job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"

mapfile -t FRAMES < <(
  python - "$SEQUENCE" <<'PY'
import sys
from pathlib import Path

sequence = sys.argv[1]
root = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")

def frame_id(path: Path) -> str:
    return path.stem.split("_")[-1]

cg = {frame_id(path) for path in (root / sequence / "cg" / "15fps").glob("*.ply")}
he = {frame_id(path) for path in (root / sequence / "he" / "15fps").glob("*.ply")}
for frame in sorted(cg & he):
    print(frame)
PY
)

srun python -u scripts/run_pudm_selected_frames.py \
  --sequence "$SEQUENCE" \
  --frames "${FRAMES[@]}" \
  --dataset-root /gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC \
  --method-name "$METHOD_NAME" \
  --checkpoint third_party/enhancement/PUDM/pointnet2/pkls/pu1k.pkl \
  --config third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json \
  --chunk-size 8192 \
  --up-rate 4 \
  --step 30 \
  --gamma 0.5
