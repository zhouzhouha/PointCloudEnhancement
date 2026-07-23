#!/bin/bash
#SBATCH --job-name=he_normals_k20
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11%5
#SBATCH --output=logs/he_normals_k20_%A_%a.out
#SBATCH --error=logs/he_normals_k20_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
SEQUENCES=(BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife)
CACHE_ROOT="${PCE_FLASH_ROOT}/data_cache/uvg_cwi_dqpc_normals"
srun "${PCE_PYTHON[@]}" -u scripts/cache_he_ground_truth_normals.py \
  --sequence "${SEQUENCES[$SLURM_ARRAY_TASK_ID]}" \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --neighbors 20
