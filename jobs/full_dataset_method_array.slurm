#!/bin/bash
#SBATCH --job-name=pce_full
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/full_dataset_%x_%A_%a.out
#SBATCH --error=logs/full_dataset_%x_%A_%a.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs

SEQUENCES=(
  BlueSpeech
  BlueVolley
  BouncingBlue
  FitFluencer
  GoodVision
  Mannequin
  OrangeKettlebell
  PinkNoir
  TicTacToe
  TrumanShow
  VictoryHeart
  VirtualLife
)

if [ -z "${METHOD:-}" ]; then
  echo "METHOD environment variable is required" >&2
  exit 2
fi

SEQUENCE="${SEQUENCES[$SLURM_ARRAY_TASK_ID]}"

module purge
module load 2023
module load CUDA/12.1.1

source /sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate torch_env

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/torch_extensions
export TMPDIR=/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/tmp
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

if [ "$METHOD" = "p2p_bridge" ]; then
  export PYTHONPATH=/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/P2P-Bridge/python_deps:/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/P2P-Bridge:/gpfs/home2/xzhou/code/PointCloudEnhancement/third_party/enhancement/P2P-Bridge/third_party/openpoints/cpp/pointnet2_batch:${PYTHONPATH:-}
  pushd third_party/enhancement/P2P-Bridge/third_party/openpoints/cpp/pointnet2_batch
  python setup.py build_ext --inplace
  popd
fi

srun python -u scripts/run_full_dataset_method.py \
  --method "$METHOD" \
  --sequence "$SEQUENCE" \
  --dataset-root /gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC \
  --results-root /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results
