#!/bin/bash
#SBATCH --job-name=pathnet_texture
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7%2
#SBATCH --output=logs/pathnet_texture_%A_%a.out
#SBATCH --error=logs/pathnet_texture_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"

SEQUENCES=(BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin OrangeKettlebell PinkNoir)
SEQUENCE="${SEQUENCES[$SLURM_ARRAY_TASK_ID]}"
mapfile -t FRAMES < <("${PCE_PYTHON[@]}" - <<PY
from pathlib import Path
root = Path('${PCE_RESULTS_ROOT}') / 'method_outputs' / 'pathnet_chunked_full' / '${SEQUENCE}' / '15fps'
print('\n'.join(sorted(path.stem.rsplit('_', 1)[-1] for path in root.glob('*.ply'))))
PY
)

srun "${PCE_PYTHON[@]}" -u scripts/run_texture_perceptual_metrics.py \
  --sequence "${SEQUENCE}" \
  --frames "${FRAMES[@]}" \
  --methods pathnet_chunked_full \
  --dataset-root "${PCE_DATASET_ROOT}" \
  --results-root "${PCE_RESULTS_ROOT}" \
  --run-name "pathnet_chunked_full/run_${SLURM_ARRAY_JOB_ID}" \
  --image-size 512 \
  --tmp-dir "${TMPDIR}" \
  --skip-pcqm
