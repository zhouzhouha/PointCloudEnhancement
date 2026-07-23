#!/bin/bash
#SBATCH --job-name=pce_cuda_smoke
#SBATCH --account=project_465003117
#SBATCH --partition=lumid
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke_lumi_cuda_environment_%j.out
#SBATCH --error=logs/smoke_lumi_cuda_environment_%j.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init nvidia
cd "${PCE_REPO_ROOT}"

srun "${PCE_PYTHON[@]}" - <<'PY'
import lpips
import numpy
import plyfile
import scipy
import skimage
import sklearn
import torch
import torchvision
import trimesh

assert torch.cuda.is_available(), "CUDA is not visible inside the container"
x = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
y = x @ x
torch.cuda.synchronize()

print("cuda smoke: OK")
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("skimage", skimage.__version__)
print("sklearn", sklearn.__version__)
print("torchvision", torchvision.__version__)
print("trimesh", trimesh.__version__)
print("matmul_sum", y.sum().item())
PY
