#!/bin/bash
#SBATCH --job-name=batchpcqm
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-251%2
#SBATCH --output=logs/batchpcqm_%A_%a.out
#SBATCH --error=logs/batchpcqm_%A_%a.err

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

if [[ ! -d "${PRED_DIR}" ]] || ! find "${PRED_DIR}" -maxdepth 1 -name '*.ply' -print -quit | grep -q .; then
  echo "Missing immutable PLY outputs in ${PRED_DIR}" >&2
  exit 2
fi

export PCE_PCQM_BIN="${PCE_REPO_ROOT}/third_party/metrics/PCQM/build_lumi_container/PCQM"
export SINGULARITYENV_PCE_PCQM_BIN="${PCE_PCQM_BIN}"
srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_pcqm_metrics.py \
  --sequence "${SEQUENCE}" \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}" \
  --method "${METHOD_NAME}" --run-id "run_${SLURM_ARRAY_JOB_ID}" \
  --tmp-dir "${TMPDIR}"
