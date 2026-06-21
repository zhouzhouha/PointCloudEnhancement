#!/bin/bash
#SBATCH --job-name=pcn_ok170_merge
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pointcleannet_ok_full170_merge_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

python -u scripts/merge_pointcleannet_full170_chunks.py \
  --sequence OrangeKettlebell \
  --source-prefix pointcleannet_full170_chunk_ \
  --target-method pointcleannet_full170 \
  --chunks 17
