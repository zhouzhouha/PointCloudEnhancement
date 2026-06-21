#!/bin/bash
#SBATCH --job-name=pdflow_ok170
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pdflow_ok_full170_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0

FRAMES=$(python - <<'PY'
print(" ".join(f"{i:04d}" for i in range(170)))
PY
)

srun python -u scripts/run_pdflow_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames ${FRAMES} \
  --method-name pdflow_full170 \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results \
  --patch-size 1024 \
  --niters 1 \
  --cluster-size 10000
