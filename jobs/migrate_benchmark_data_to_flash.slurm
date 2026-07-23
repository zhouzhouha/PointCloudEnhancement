#!/bin/bash
#SBATCH --job-name=pce_to_flash
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pce_to_flash_%j.out
#SBATCH --error=logs/pce_to_flash_%j.err

set -euo pipefail

SCRATCH_ROOT=/scratch/project_465003117
FLASH_ROOT=/flash/project_465003117/PointCloudEnhancement
DATA_SRC=${SCRATCH_ROOT}/data/
DATA_DST=${FLASH_ROOT}/data/
RESULTS_SRC=${SCRATCH_ROOT}/PointCloudEnhancement/full_dataset/results/
RESULTS_DST=${FLASH_ROOT}/results/

mkdir -p "${DATA_DST}" "${RESULTS_DST}"
# Never overwrite an existing named artifact. The verification pass fails if
# an existing flash file differs from its scratch source. Flash may contain
# additional results written there directly by newer jobs.
rsync -a --ignore-existing "${DATA_SRC}" "${DATA_DST}"
rsync -a --ignore-existing "${RESULTS_SRC}" "${RESULTS_DST}"

DATA_DIFF=$(rsync -ani --delete "${DATA_SRC}" "${DATA_DST}")
RESULTS_DIFF=$(rsync -ani "${RESULTS_SRC}" "${RESULTS_DST}")
if [[ -n "${DATA_DIFF}" || -n "${RESULTS_DIFF}" ]]; then
  printf 'Dataset verification differences:\n%s\n' "${DATA_DIFF}" >&2
  printf 'Result verification differences:\n%s\n' "${RESULTS_DIFF}" >&2
  exit 1
fi

find "${SCRATCH_ROOT}/data" -depth -delete
ln -s "${FLASH_ROOT}/data" "${SCRATCH_ROOT}/data"
find "${SCRATCH_ROOT}/PointCloudEnhancement/full_dataset/results" -depth -delete
ln -s "${FLASH_ROOT}/results" "${SCRATCH_ROOT}/PointCloudEnhancement/full_dataset/results"

echo "Verified benchmark dataset and results in ${FLASH_ROOT}"
