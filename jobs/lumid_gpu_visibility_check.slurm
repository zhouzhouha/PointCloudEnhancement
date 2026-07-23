#!/bin/bash
#SBATCH --job-name=pce_gpu_check
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/lumid_gpu_check_%j.out
#SBATCH --error=logs/lumid_gpu_check_%j.err

set -euo pipefail

echo "node=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
srun "${PCE_PYTHON[@]}" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
