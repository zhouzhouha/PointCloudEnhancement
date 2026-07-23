#!/bin/bash
#SBATCH --job-name=batchnorm_rocm
#SBATCH --account=project_465003117
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-251%5
#SBATCH --output=logs/batchnorm_rocm_%A_%a.out
#SBATCH --error=logs/batchnorm_rocm_%A_%a.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G

PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
FLASH_ROOT=/flash/project_465003117/PointCloudEnhancement
DATASET_ROOT=${FLASH_ROOT}/data/UVG_CWI_DQPC/UVG-CWI-DQPC
RESULTS_ROOT=${FLASH_ROOT}/results
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env
METHOD_PREFIX=${METHOD_PREFIX:?METHOD_PREFIX is required}
SEQUENCES=(
  BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin
  OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife
)
BATCHES_PER_SEQUENCE=21
SEQUENCE_INDEX=$((SLURM_ARRAY_TASK_ID / BATCHES_PER_SEQUENCE))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID % BATCHES_PER_SEQUENCE))
SEQUENCE=${SEQUENCES[$SEQUENCE_INDEX]}
METHOD_NAME=$(printf '%s_batch_%02d' "${METHOD_PREFIX}" "${BATCH_INDEX}")
PRED_DIR=${RESULTS_ROOT}/method_outputs/${METHOD_NAME}/${SEQUENCE}/15fps
[[ -d "${PRED_DIR}" ]] || { echo "Missing immutable output: ${PRED_DIR}" >&2; exit 2; }

TMP_ROOT=/scratch/project_465003117/PointCloudEnhancement/tmp/${SLURM_JOB_ID}
mkdir -p "${TMP_ROOT}"
cd "${REPO_ROOT}"
srun singularity exec \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "/flash/project_465003117:/flash/project_465003117" \
  --bind "/scratch/project_465003117:/scratch/project_465003117" \
  --env "TMPDIR=${TMP_ROOT}" \
  "${SIF}" "${VENV}/bin/python" -u scripts/run_pathnet_missing_normal_metrics.py \
    --sequence "${SEQUENCE}" --batch-index 0 --batch-size 10 \
    --dataset-root "${DATASET_ROOT}" --results-root "${RESULTS_ROOT}" \
    --method "${METHOD_NAME}" \
    --normal-cache-root "${FLASH_ROOT}/data_cache/uvg_cwi_dqpc_normals" \
    --normal-k 20 --run-id "run_${SLURM_ARRAY_JOB_ID}_pca_knn20"
