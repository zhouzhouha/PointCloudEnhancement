#!/bin/bash
#SBATCH --job-name=sapcu_lumid_smoke
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sapcu_lumid_smoke_%j.out
#SBATCH --error=logs/sapcu_lumid_smoke_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"

METHOD="sapcu_4x_2048_lumid_smoke_20260721"
TARGET="${PCE_RESULTS_ROOT}/method_outputs/${METHOD}"
if [[ -e "${TARGET}" ]]; then
  echo "Refusing to overwrite ${TARGET}" >&2
  exit 2
fi

srun "${PCE_PYTHON[@]}" -u scripts/run_sapcu_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --results-root "${PCE_RESULTS_ROOT}" \
  --method-name "${METHOD}" \
  --input-points 2048 \
  --output-points 8192
