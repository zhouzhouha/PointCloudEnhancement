#!/bin/bash
#SBATCH --job-name=batchtex
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-251%2
#SBATCH --output=logs/batchtex_%A_%a.out
#SBATCH --error=logs/batchtex_%A_%a.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"

METHOD_PREFIX=${METHOD_PREFIX:?METHOD_PREFIX is required}
SEQUENCES=(
  BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin
  OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife
)
BATCHES_PER_SEQUENCE=21
SEQUENCE_INDEX=$((SLURM_ARRAY_TASK_ID / BATCHES_PER_SEQUENCE))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % BATCHES_PER_SEQUENCE))
SEQUENCE=${SEQUENCES[$SEQUENCE_INDEX]}
METHOD_NAME=$(printf '%s_batch_%02d' "${METHOD_PREFIX}" "${BATCH_INDEX}")
PRED_DIR="${PCE_RESULTS_ROOT}/method_outputs/${METHOD_NAME}/${SEQUENCE}/15fps"

if [[ ! -d "${PRED_DIR}" ]]; then
  echo "Missing immutable method output directory: ${PRED_DIR}" >&2
  exit 2
fi

mapfile -t FRAMES < <("${PCE_PYTHON[@]}" - <<PY
from pathlib import Path
root = Path('${PRED_DIR}')
print('\n'.join(sorted(path.stem.rsplit('_', 1)[-1] for path in root.glob('*.ply'))))
PY
)
if [[ ${#FRAMES[@]} -eq 0 ]]; then
  echo "No immutable PLY outputs in ${PRED_DIR}" >&2
  exit 2
fi

srun "${PCE_PYTHON[@]}" -u scripts/run_texture_perceptual_metrics.py \
  --sequence "${SEQUENCE}" --frames "${FRAMES[@]}" --methods "${METHOD_NAME}" \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}" \
  --run-name "${METHOD_NAME}/run_${SLURM_ARRAY_JOB_ID}" \
  --image-size 512 --tmp-dir "${TMPDIR}" --skip-pcqm
