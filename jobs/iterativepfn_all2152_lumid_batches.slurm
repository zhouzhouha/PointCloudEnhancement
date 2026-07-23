#!/bin/bash
#SBATCH --job-name=itpfn2152
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-251%2
#SBATCH --output=logs/iterativepfn2152_%A_%a.out
#SBATCH --error=logs/iterativepfn2152_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"

SEQUENCES=(
  BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin
  OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife
)
BATCH_SIZE=10
BATCHES_PER_SEQUENCE=21
METHOD_PREFIX=${METHOD_PREFIX:-iterativepfn_full_20260722}
SEQUENCE_INDEX=$((SLURM_ARRAY_TASK_ID / BATCHES_PER_SEQUENCE))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % BATCHES_PER_SEQUENCE))
SEQUENCE=${SEQUENCES[$SEQUENCE_INDEX]}
METHOD_NAME=$(printf '%s_batch_%02d' "${METHOD_PREFIX}" "${BATCH_INDEX}")

mapfile -t FRAMES < <("${PCE_PYTHON[@]}" - "${PCE_DATASET_ROOT}" "${SEQUENCE}" "${BATCH_INDEX}" "${BATCH_SIZE}" <<'PY'
from pathlib import Path
import sys
root, sequence, batch_index, batch_size = sys.argv[1:]
batch_index, batch_size = int(batch_index), int(batch_size)
cg = {p.stem.rsplit('_', 1)[-1] for p in (Path(root) / sequence / 'cg' / '15fps').glob('*.ply')}
he = {p.stem.rsplit('_', 1)[-1] for p in (Path(root) / sequence / 'he' / '15fps').glob('*.ply')}
print('\n'.join(sorted(cg & he)[batch_index * batch_size:(batch_index + 1) * batch_size]))
PY
)
if [[ ${#FRAMES[@]} -eq 0 ]]; then
  echo "No paired frames for ${SEQUENCE} batch ${BATCH_INDEX}; nothing to run."
  exit 0
fi

TARGET="${PCE_RESULTS_ROOT}/method_outputs/${METHOD_NAME}/${SEQUENCE}/15fps"
if [[ -e "${TARGET}" ]]; then
  echo "Refusing to overwrite immutable batch output: ${TARGET}" >&2
  exit 2
fi

srun "${PCE_PYTHON[@]}" -u scripts/run_full_dataset_method.py \
  --method iterativepfn --method-name "${METHOD_NAME}" \
  --sequence "${SEQUENCE}" --frames "${FRAMES[@]}" \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}"
