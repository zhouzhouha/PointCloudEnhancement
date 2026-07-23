#!/bin/bash
#SBATCH --job-name=pathnet2152_pcqm
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/pathnet2152_pcqm_%A_%a.out
#SBATCH --error=logs/pathnet2152_pcqm_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
METHOD=pathnet_chunked_full_all2152_20260721
SEQUENCES=(BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife)
export PCE_PCQM_BIN="${PCE_REPO_ROOT}/third_party/metrics/PCQM/build_lumi_container/PCQM"
export SINGULARITYENV_PCE_PCQM_BIN="${PCE_PCQM_BIN}"
srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_pcqm_metrics.py \
  --sequence "${SEQUENCES[$SLURM_ARRAY_TASK_ID]}" \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}" \
  --method "${METHOD}" --run-id "run_${SLURM_ARRAY_JOB_ID}" --tmp-dir "${TMPDIR}"
