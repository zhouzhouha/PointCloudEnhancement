#!/bin/bash
#SBATCH --job-name=pathnet2152_tables
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pathnet2152_tables_%j.out
#SBATCH --error=logs/pathnet2152_tables_%j.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
METHOD=pathnet_chunked_full_all2152_20260721
srun "${PCE_PYTHON[@]}" -u scripts/build_pathnet_paper_tables.py \
  --results-root "${PCE_RESULTS_ROOT}" --method "${METHOD}" \
  --normal-run run_20060548_pca_knn20 \
  --texture-run run_20060549 \
  --pcqm-run run_20060550 \
  --out-dir "${PCE_FLASH_ROOT}/paper_artifacts/${METHOD}" --expected-frames 2152
