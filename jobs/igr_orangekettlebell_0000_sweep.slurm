#!/bin/bash
#SBATCH --job-name=igr_ok_0000_sweep
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/igr_ok_0000_sweep_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

python -u scripts/run_igr_parameter_sweep.py \
  --sequence OrangeKettlebell \
  --frame 0000 \
  --sample-points 200000
