#!/bin/bash
#SBATCH --job-name=gqenet_rocm_deps
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gqenet_rocm_deps_%j.out
#SBATCH --error=logs/gqenet_rocm_deps_%j.err

set -euo pipefail
module --force purge
module load LUMI/25.03 partition/C
PROJECT_ROOT=/project/project_465003117
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
VENV=${PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_rocm_torch_env
singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" \
  "${VENV}/bin/python" -m pip install plyfile
singularity exec --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" "${SIF}" \
  "${VENV}/bin/python" -c 'import plyfile; print("plyfile ready")'
