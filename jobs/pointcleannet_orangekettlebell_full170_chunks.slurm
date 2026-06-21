#!/bin/bash
#SBATCH --job-name=pcn_ok170
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --array=0-16%4
#SBATCH --output=logs/pointcleannet_ok_full170_chunk_%A_%a.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

FRAMES=$(python - <<'PY'
import os
task = int(os.environ["SLURM_ARRAY_TASK_ID"])
start = task * 10
end = min(170, start + 10)
print(" ".join(f"{i:04d}" for i in range(start, end)))
PY
)

python -u scripts/run_pointcleannet_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames ${FRAMES} \
  --method-name pointcleannet_full170_chunk_${SLURM_ARRAY_TASK_ID} \
  --nrun 3 \
  --workers 1 \
  --cache-capacity 1
