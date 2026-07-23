#!/bin/bash
#SBATCH --job-name=pathnet_pcqm
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7%2
#SBATCH --output=logs/pathnet_pcqm_%A_%a.out
#SBATCH --error=logs/pathnet_pcqm_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
SEQUENCES=(BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin OrangeKettlebell PinkNoir)
export PCE_PCQM_BIN="${PCE_REPO_ROOT}/third_party/metrics/PCQM/build_lumi_container/PCQM"
export SINGULARITYENV_PCE_PCQM_BIN="${PCE_PCQM_BIN}"
EXTRA_ARGS=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  EXTRA_ARGS+=(--max-frames "${MAX_FRAMES}")
fi
srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_pcqm_metrics.py \
  --sequence "${SEQUENCES[$SLURM_ARRAY_TASK_ID]}" \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}" \
  --method pathnet_chunked_full --run-id "run_${SLURM_ARRAY_JOB_ID}" \
  --tmp-dir "${TMPDIR}" "${EXTRA_ARGS[@]}"
