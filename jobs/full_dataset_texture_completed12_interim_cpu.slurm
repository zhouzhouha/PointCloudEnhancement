#!/bin/bash
#SBATCH --job-name=pce_tex_c12_cpu
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/full_dataset_texture_completed12_cpu_%A_%a.out
#SBATCH --error=logs/full_dataset_texture_completed12_cpu_%A_%a.err

set -euo pipefail

source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
mkdir -p logs

SEQUENCES=(
  BlueSpeech
  BlueVolley
  BouncingBlue
  FitFluencer
  GoodVision
  Mannequin
  OrangeKettlebell
  PinkNoir
  TicTacToe
  TrumanShow
  VictoryHeart
  VirtualLife
)
SEQUENCE="${SEQUENCES[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

srun "${PCE_PYTHON[@]}" -u scripts/run_texture_perceptual_metrics.py \
  --sequence "$SEQUENCE" \
  --frames $("${PCE_PYTHON[@]}" - <<PY
from pathlib import Path
root = Path('${PCE_DATASET_ROOT}') / '$SEQUENCE'
cg = {p.stem.split('_')[-1] for p in (root / 'cg' / '15fps').glob('*.ply')}
he = {p.stem.split('_')[-1] for p in (root / 'he' / '15fps').glob('*.ply')}
print(' '.join(sorted(cg & he)))
PY
) \
  --methods \
    apuldi_local_pu1k_4x_2048_full \
    neuralpoints_16x_2048_full \
    pucrn_pu1k_4x_full \
    pufm_pugan_4x_full \
    puflow_discrete_full \
    crcir_aftercomp_4x_full \
    upsample_clean_ounet_full \
    spu_pointnet_4x_full \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --run-name full_dataset_completed12_interim \
  --image-size 512 \
  --tmp-dir "${TMPDIR}"
