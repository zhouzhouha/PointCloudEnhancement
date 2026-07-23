#!/bin/bash
#SBATCH --job-name=pcqm_container
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_lumi_pcqm_container_%j.out
#SBATCH --error=logs/build_lumi_pcqm_container_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/C

PROJECT_ROOT=/project/project_465003117
REPO_ROOT="${PROJECT_ROOT}/xuemei/PointCloudEnhancement"
PCQM_ROOT="${REPO_ROOT}/third_party/metrics/PCQM"
BUILD_ROOT="${PCQM_ROOT}/build_lumi_container"
CONTAINER="${PROJECT_ROOT}/containers/pce-pytorch-2.4.1-cu121-devel.sif"

test -f "${PCQM_ROOT}/CMakeLists.txt"
test -r "${CONTAINER}"
test ! -e "${BUILD_ROOT}/PCQM"

singularity exec --cleanenv --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${CONTAINER}" \
  bash -lc "cmake -S '${PCQM_ROOT}' -B '${BUILD_ROOT}' -DCMAKE_BUILD_TYPE=Release && cmake --build '${BUILD_ROOT}' --parallel '${SLURM_CPUS_PER_TASK:-8}'"

test -x "${BUILD_ROOT}/PCQM"
echo "PCQM container-compatible build: ${BUILD_ROOT}/PCQM"
