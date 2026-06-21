#!/bin/bash
#SBATCH --job-name=sal_ok_0000_smoke
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/sal_ok_0000_smoke_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

python -u scripts/prepare_uvg_frame_for_sal.py \
  --sequence OrangeKettlebell \
  --frame 0000 \
  --max-points 50000

cd third_party/SCUTSurface/reconstruction/SAL/code

srun python -u training/exp_runner.py \
  --batch_size 1 \
  --nepoch 5 \
  --conf confs/orangekettlebell_frame_0000.conf \
  --workers 1 \
  --gpu all
