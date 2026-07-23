#!/bin/bash
#SBATCH --job-name=pce_pcqm
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_lumi_pcqm_%j.out
#SBATCH --error=logs/build_lumi_pcqm_%j.err

set -euo pipefail

REPO_ROOT=/project/project_465003117/xuemei/PointCloudEnhancement
PCQM_ROOT="${REPO_ROOT}/third_party/metrics/PCQM"

module --force purge
module load LUMI/25.03 partition/C

if [[ ! -f "${PCQM_ROOT}/CMakeLists.txt" ]]; then
  echo "PCQM source is missing at ${PCQM_ROOT}" >&2
  exit 2
fi

cmake -S "${PCQM_ROOT}" -B "${PCQM_ROOT}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${PCQM_ROOT}/build" --parallel "${SLURM_CPUS_PER_TASK:-8}"
test -x "${PCQM_ROOT}/build/PCQM"
"${PCQM_ROOT}/build/PCQM" 2>&1 || true
echo "PCQM build completed: ${PCQM_ROOT}/build/PCQM"
