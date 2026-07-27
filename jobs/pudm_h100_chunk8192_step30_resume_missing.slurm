#!/bin/bash
#SBATCH --job-name=pudm8192_resume
#SBATCH --account=mcsei7083
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/pudm_h100_chunk8192_step30_resume_%A_%a.out
#SBATCH --error=logs/pudm_h100_chunk8192_step30_resume_%A_%a.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs

SEQUENCES=(
  FitFluencer
  GoodVision
  Mannequin
)

SEQUENCE="${SEQUENCES[$SLURM_ARRAY_TASK_ID]}"
METHOD_NAME=pudm_pu1k_4x_chunk8192_step30_full
PROJECT_RESULTS=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results
DATASET_ROOT=/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC

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

mapfile -t FRAMES < <(
  python - "$SEQUENCE" "$METHOD_NAME" "$DATASET_ROOT" "$PROJECT_RESULTS" <<'PY'
import re
import sys
from pathlib import Path

sequence, method_name, dataset_root, results_root = sys.argv[1:]
dataset_root = Path(dataset_root)
results_root = Path(results_root)

def frame_id(path: Path) -> str:
    return path.stem.split("_")[-1]

cg = {frame_id(path) for path in (dataset_root / sequence / "cg" / "15fps").glob("*.ply")}
he = {frame_id(path) for path in (dataset_root / sequence / "he" / "15fps").glob("*.ply")}
done = set()
out_dir = results_root / "method_outputs" / method_name / sequence
for path in out_dir.rglob("*.ply") if out_dir.exists() else []:
    match = re.search(r"(?:frame_)?(\d{4})", path.stem)
    if match:
        done.add(match.group(1))
for frame in sorted((cg & he) - done):
    print(frame)
PY
)

if [ "${#FRAMES[@]}" -eq 0 ]; then
  echo "[pudm-resume] ${SEQUENCE} already complete; no duplicate work."
  exit 0
fi

echo "[pudm-resume] method=${METHOD_NAME} sequence=${SEQUENCE} frames=${#FRAMES[@]} first=${FRAMES[0]} last=${FRAMES[-1]} job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"

srun python -u scripts/run_pudm_selected_frames.py \
  --sequence "$SEQUENCE" \
  --frames "${FRAMES[@]}" \
  --dataset-root "$DATASET_ROOT" \
  --method-name "$METHOD_NAME" \
  --checkpoint third_party/enhancement/PUDM/pointnet2/pkls/pu1k.pkl \
  --config third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json \
  --chunk-size 8192 \
  --up-rate 4 \
  --step 30 \
  --gamma 0.5
