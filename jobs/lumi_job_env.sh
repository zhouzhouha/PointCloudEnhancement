#!/bin/bash

# Shared LUMI paths and NVIDIA CUDA container setup for full-dataset jobs.
# This file is sourced by the SLURM templates; it is not submitted directly.

pce_lumi_init() {
  local accelerator="${1:?usage: pce_lumi_init nvidia|cpu}"

  export PCE_PROJECT_ROOT="${PCE_PROJECT_ROOT:-/project/project_465003117}"
  export PCE_SCRATCH_ROOT="${PCE_SCRATCH_ROOT:-/scratch/project_465003117}"
  export PCE_FLASH_ROOT="${PCE_FLASH_ROOT:-/flash/project_465003117/PointCloudEnhancement}"
  export PCE_REPO_ROOT="${PCE_REPO_ROOT:-${PCE_PROJECT_ROOT}/xuemei/PointCloudEnhancement}"
  export PCE_DATASET_ROOT="${PCE_DATASET_ROOT:-${PCE_FLASH_ROOT}/data/UVG_CWI_DQPC/UVG-CWI-DQPC}"
  export PCE_FULL_DATASET_ROOT="${PCE_FULL_DATASET_ROOT:-${PCE_SCRATCH_ROOT}/PointCloudEnhancement/full_dataset}"
  export PCE_RESULTS_ROOT="${PCE_RESULTS_ROOT:-${PCE_FLASH_ROOT}/results}"
  export PCE_CONTAINER="${PCE_CONTAINER:-${PCE_PROJECT_ROOT}/containers/pce-pytorch-2.4.1-cu121-devel.sif}"
  export PCE_VENV="${PCE_VENV:-${PCE_PROJECT_ROOT}/envs/PointCloudEnhancement/lumi_cuda_torch_env}"

  if [[ ! -d "${PCE_REPO_ROOT}" ]]; then
    echo "Repository not found: ${PCE_REPO_ROOT}" >&2
    return 2
  fi
  if [[ ! -r "${PCE_CONTAINER}" ]]; then
    echo "LUMI PyTorch container not readable: ${PCE_CONTAINER}" >&2
    return 2
  fi

  module --force purge
  if [[ "${accelerator}" == "nvidia" ]]; then
    module load LUMI/24.03 partition/D CUDA/12.2.2
  elif [[ "${accelerator}" == "cpu" ]]; then
    module load LUMI/25.03 partition/C
  else
    echo "Unknown accelerator '${accelerator}'; expected nvidia or cpu" >&2
    return 2
  fi

  export TMPDIR="${PCE_TMPDIR:-${PCE_FULL_DATASET_ROOT}/tmp}"
  export TORCH_EXTENSIONS_DIR="${PCE_TORCH_EXTENSIONS_DIR:-${PCE_FULL_DATASET_ROOT}/torch_extensions}"
  export TORCH_HOME="${PCE_TORCH_HOME:-${PCE_FULL_DATASET_ROOT}/torch_cache}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
  export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
  mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${TORCH_HOME}" "${PCE_RESULTS_ROOT}"

  local -a singularity_args=(exec --cleanenv)
  if [[ "${accelerator}" == "nvidia" ]]; then
    singularity_args+=(--nv)
  fi
  singularity_args+=(
    --bind "${PCE_PROJECT_ROOT}:${PCE_PROJECT_ROOT}"
    --bind "${PCE_SCRATCH_ROOT}:${PCE_SCRATCH_ROOT}"
    --bind "/flash/project_465003117:/flash/project_465003117"
    --env "TMPDIR=${TMPDIR}"
    --env "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
    --env "TORCH_HOME=${TORCH_HOME}"
    --env "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    --env "MAX_JOBS=${MAX_JOBS}"
    --env "OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
  )

  # The overlay venv should be created inside this container with
  # --system-site-packages so it reuses the container's CUDA-enabled PyTorch.
  local python_exe=/opt/conda/bin/python
  # Container-created venvs may use an absolute /opt/conda Python symlink that
  # is intentionally broken on the host but valid after entering the image.
  if [[ -f "${PCE_VENV}/pyvenv.cfg" ]]; then
    singularity_args+=(
      --env "VIRTUAL_ENV=${PCE_VENV}"
      --env "PATH=${PCE_VENV}/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"
    )
    python_exe="${PCE_VENV}/bin/python"
  else
    echo "WARNING: PCE_VENV not found at ${PCE_VENV}; using base container packages only." >&2
  fi

  PCE_PYTHON=(singularity "${singularity_args[@]}" "${PCE_CONTAINER}" "${python_exe}")
  export PCE_PROJECT_ROOT PCE_SCRATCH_ROOT PCE_FLASH_ROOT PCE_REPO_ROOT PCE_DATASET_ROOT
  export PCE_FULL_DATASET_ROOT PCE_RESULTS_ROOT PCE_CONTAINER PCE_VENV
}
