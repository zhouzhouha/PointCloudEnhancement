#!/bin/bash
#SBATCH --job-name=gf_sor_ok_sel10
#SBATCH --partition=cbuild
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/geometry_filter_sor_ok_selected10_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

python -u scripts/run_geometry_filter_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 0010 0020 0030 0040 0050 0060 0070 0080 0090 \
  --method-name geometry_filter_sor \
  --sor-neighbors 20 \
  --sor-std-ratio 2.0 \
  --radius 0 \
  --voxel-size 0 \
  --color-knn 1
