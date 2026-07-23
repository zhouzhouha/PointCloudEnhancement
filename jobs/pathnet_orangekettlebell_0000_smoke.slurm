#!/bin/bash
#SBATCH --job-name=pathnet_ok0
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pathnet_ok_0000_smoke_%j.out
#SBATCH --error=logs/pathnet_ok_0000_smoke_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"

srun "${PCE_PYTHON[@]}" -u scripts/run_pathnet_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --results-root "${PCE_RESULTS_ROOT}" \
  --method-name pathnet_chunked_lumid_smoke \
  --chunk-size 50000 \
  --batch-size 512 \
  --knn 128 \
  --iterations 2
