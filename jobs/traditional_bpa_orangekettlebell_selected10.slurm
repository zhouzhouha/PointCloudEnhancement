#!/bin/bash
#SBATCH --job-name=trad_bpa_ok10
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/traditional_bpa_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -u scripts/run_traditional_reconstruction_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --reconstruction-method bpa \
  --method-name traditional_bpa \
  --knn 30
