#!/bin/bash
#SBATCH --job-name=submit_gradpu_wave
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/submit_gradpu_wave_%j.out
#SBATCH --error=logs/submit_gradpu_wave_%j.err

set -euo pipefail

REPO_ROOT=/project/project_465003117/xuemei/PointCloudEnhancement
NEXT_ARRAY=${NEXT_ARRAY:?NEXT_ARRAY is required, for example 12-16}
METHOD_PREFIX=${METHOD_PREFIX:?METHOD_PREFIX is required}
cd "${REPO_ROOT}"

INFERENCE_JOB_ID=$(sbatch --parsable \
  --array="${NEXT_ARRAY}%2" \
  --export="ALL,METHOD_PREFIX=${METHOD_PREFIX}" \
  jobs/gradpu_all2152_lumid_batches.slurm)
echo "submitted Grad-PU inference array ${INFERENCE_JOB_ID} for ${NEXT_ARRAY}"

NORMAL_JOB_ID=$(sbatch --parsable \
  --array="${NEXT_ARRAY}%5" \
  --dependency="aftercorr:${INFERENCE_JOB_ID}" \
  --export="ALL,METHOD_PREFIX=${METHOD_PREFIX}" \
  jobs/batched_method_normal_k20_array.slurm)
echo "submitted matching k=20 normal array ${NORMAL_JOB_ID} aftercorr:${INFERENCE_JOB_ID}"

TEXTURE_JOB_ID=$(sbatch --parsable \
  --array="${NEXT_ARRAY}%1" \
  --dependency="aftercorr:${INFERENCE_JOB_ID}" \
  --export="ALL,METHOD_PREFIX=${METHOD_PREFIX}" \
  jobs/batched_method_texture_array.slurm)
echo "submitted texture/perceptual array ${TEXTURE_JOB_ID} aftercorr:${INFERENCE_JOB_ID}"

PCQM_JOB_ID=$(sbatch --parsable \
  --array="${NEXT_ARRAY}%1" \
  --dependency="aftercorr:${TEXTURE_JOB_ID}" \
  --export="ALL,METHOD_PREFIX=${METHOD_PREFIX}" \
  jobs/batched_method_pcqm_array.slurm)
echo "submitted PCQM array ${PCQM_JOB_ID} aftercorr:${TEXTURE_JOB_ID}"
