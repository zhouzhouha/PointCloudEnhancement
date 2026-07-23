#!/bin/bash
#SBATCH --job-name=pce_full
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/full_dataset_%x_%A_%a.out
#SBATCH --error=logs/full_dataset_%x_%A_%a.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"
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

if [ "$METHOD" = "p2p_bridge" ]; then
  export PYTHONPATH="${PCE_REPO_ROOT}/third_party/enhancement/P2P-Bridge/python_deps:${PCE_REPO_ROOT}/third_party/enhancement/P2P-Bridge:${PCE_REPO_ROOT}/third_party/enhancement/P2P-Bridge/third_party/openpoints/cpp/pointnet2_batch:${PYTHONPATH:-}"
  export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
  pushd third_party/enhancement/P2P-Bridge/third_party/openpoints/cpp/pointnet2_batch
  "${PCE_PYTHON[@]}" setup.py build_ext --inplace
  popd
fi

srun "${PCE_PYTHON[@]}" -u scripts/run_full_dataset_method.py \
  --method "$METHOD" \
  --sequence "$SEQUENCE" \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --results-root "${PCE_RESULTS_ROOT}"
