#!/bin/bash
#SBATCH --job-name=pathnet_merge2152
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pathnet_merge2152_%j.out
#SBATCH --error=logs/pathnet_merge2152_%j.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"

srun "${PCE_PYTHON[@]}" -u scripts/consolidate_pathnet_full_dataset.py \
  --results-root "${PCE_RESULTS_ROOT}" \
  --method pathnet_chunked_full_all2152_20260721 \
  --batch-prefixes pathnet_chunked_full_batch_ pathnet_chunked_full_completion_20260721_batch_ \
  --expected-total 2152
