#!/bin/bash
#SBATCH --job-name=pcqm_rocm_build
#SBATCH --account=project_465003117
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=mi250:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcqm_rocm_build_%j.out
#SBATCH --error=logs/pcqm_rocm_build_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/G

PROJECT_ROOT=/project/project_465003117
REPO_ROOT=${PROJECT_ROOT}/xuemei/PointCloudEnhancement
SOURCE=${REPO_ROOT}/third_party/metrics/PCQM
BUILD=${SOURCE}/build_lumi_rocm
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif

if [[ -e "${BUILD}" ]]; then
  echo "Refusing to overwrite existing ROCm-container PCQM build: ${BUILD}" >&2
  exit 2
fi

singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" \
  /bin/bash -lc "set -euo pipefail; cmake -S '${SOURCE}' -B '${BUILD}' -DCMAKE_BUILD_TYPE=Release; cmake --build '${BUILD}' --parallel 4; linked=\$(ldd '${BUILD}/PCQM'); echo \"\${linked}\"; if grep -q 'not found' <<<\"\${linked}\"; then exit 1; fi"

[[ -x "${BUILD}/PCQM" ]]
echo "PCQM_ROCM_BUILD=PASS path=${BUILD}/PCQM"
