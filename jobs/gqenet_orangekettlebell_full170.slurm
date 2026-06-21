#!/bin/bash
#SBATCH --job-name=gqe_ok170
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gqenet_ok_full170_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

FRAMES=$(python - <<'PY'
print(" ".join(f"{i:04d}" for i in range(170)))
PY
)

srun python -u scripts/run_gqenet_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames ${FRAMES} \
  --method-name gqenet_full170 \
  --test-batch-size 8
