#!/bin/bash
#SBATCH --job-name=pugauss_resume
#SBATCH --account=mcsei7083
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/pu_gaussian_resume_%A_%a.out
#SBATCH --error=logs/pu_gaussian_resume_%A_%a.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs

SEQUENCES=(
  BlueSpeech
  BlueVolley
  BouncingBlue
  OrangeKettlebell
  TicTacToe
  TrumanShow
  VictoryHeart
  VirtualLife
)

SEQUENCE="${SEQUENCES[$SLURM_ARRAY_TASK_ID]}"
METHOD_NAME=pu_gaussian_pu1k_4x_full
PROJECT_RESULTS=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results
DATASET_ROOT=/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC

module purge
module load 2023
module load CUDA/12.1.1

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/torch_extensions
export TMPDIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/tmp
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

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
  echo "[pugauss-resume] ${SEQUENCE} already complete; no duplicate work."
  exit 0
fi

echo "[pugauss-resume] method=${METHOD_NAME} sequence=${SEQUENCE} frames=${#FRAMES[@]} first=${FRAMES[0]} last=${FRAMES[-1]} job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"

srun python -u scripts/run_full_dataset_method.py \
  --method pu_gaussian \
  --sequence "$SEQUENCE" \
  --dataset-root "$DATASET_ROOT" \
  --results-root "$PROJECT_RESULTS" \
  --frames "${FRAMES[@]}"
