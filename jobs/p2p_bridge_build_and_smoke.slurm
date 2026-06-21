#!/bin/bash
#SBATCH --job-name=p2p_bridge_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/p2p_bridge_smoke_%j.out
#SBATCH --error=logs/p2p_bridge_smoke_%j.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

module purge
module load 2023
module load CUDA/12.1.1

export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH=/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/P2P-Bridge/python_deps:/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/P2P-Bridge:/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/P2P-Bridge/third_party/openpoints/cpp/pointnet2_batch

cd third_party/enhancement/P2P-Bridge/third_party/openpoints/cpp/pointnet2_batch
/home/xzhou/.conda/envs/torch_env/bin/python setup.py build_ext --inplace

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
/home/xzhou/.conda/envs/torch_env/bin/python scripts/run_p2p_bridge_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name p2p_bridge_pvds_punet \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results

mkdir -p results/method_outputs results/uvg_cwi_dqpc/OrangeKettlebell
ln -sfn /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/method_outputs/p2p_bridge_pvds_punet \
  results/method_outputs/p2p_bridge_pvds_punet
ln -sfn /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/uvg_cwi_dqpc/OrangeKettlebell/p2p_bridge_pvds_punet \
  results/uvg_cwi_dqpc/OrangeKettlebell/p2p_bridge_pvds_punet
