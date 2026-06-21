#!/bin/bash
#SBATCH --job-name=igr_ok_0000_smoke
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/igr_ok_0000_smoke_h100_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

python -u scripts/prepare_uvg_frame_for_igr.py \
  --sequence OrangeKettlebell \
  --frame 0000 \
  --max-points 50000

cd third_party/SCUTSurface/reconstruction/IGR/code

srun python -u reconstruction/run.py \
  --points_batch 8192 \
  --nepoch 100 \
  --conf orangekettlebell_frame_0000.conf \
  --expname OrangeKettlebell_frame_0000_igr100_50k \
  --gpu 0

srun python -u reconstruction/run.py \
  --eval \
  --checkpoint latest \
  --conf orangekettlebell_frame_0000.conf \
  --expname OrangeKettlebell_frame_0000_igr100_50k \
  --gpu 0

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

IGR_MESH=$(find third_party/SCUTSurface/reconstruction/IGR/exps/OrangeKettlebell_frame_0000_igr100_50k -path '*/evaluation/*/*.ply' -print | sort | tail -n 1)
python -u scripts/denormalize_igr_output.py \
  --mesh "$IGR_MESH" \
  --normalization third_party/SCUTSurface/reconstruction/IGR/data/uvg_kettlebell/points/OrangeKettlebell/frame_0000_normalization.json \
  --out-mesh results/method_outputs/igr/OrangeKettlebell/15fps/frame_0000_mesh.ply \
  --out-points results/method_outputs/igr/OrangeKettlebell/15fps/frame_0000.ply \
  --sample-points 200000 \
  --seed 0
