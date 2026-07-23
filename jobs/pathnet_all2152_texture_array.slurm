#!/bin/bash
#SBATCH --job-name=pathnet2152_tex
#SBATCH --account=project_465003117
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/pathnet2152_tex_%A_%a.out
#SBATCH --error=logs/pathnet2152_tex_%A_%a.err

set -euo pipefail
source /project/project_465003117/xuemei/PointCloudEnhancement/jobs/lumi_job_env.sh
pce_lumi_init cpu
cd "${PCE_REPO_ROOT}"
METHOD=pathnet_chunked_full_all2152_20260721
SEQUENCES=(BlueSpeech BlueVolley BouncingBlue FitFluencer GoodVision Mannequin OrangeKettlebell PinkNoir TicTacToe TrumanShow VictoryHeart VirtualLife)
SEQUENCE=${SEQUENCES[$SLURM_ARRAY_TASK_ID]}
mapfile -t FRAMES < <("${PCE_PYTHON[@]}" - <<PY
from pathlib import Path
root = Path('${PCE_RESULTS_ROOT}') / 'method_outputs' / '${METHOD}' / '${SEQUENCE}' / '15fps'
print('\n'.join(sorted(path.stem.rsplit('_', 1)[-1] for path in root.glob('*.ply'))))
PY
)
if [[ ${#FRAMES[@]} -eq 0 ]]; then
  echo "No frames found for ${SEQUENCE}" >&2
  exit 2
fi
srun "${PCE_PYTHON[@]}" -u scripts/run_texture_perceptual_metrics.py \
  --sequence "${SEQUENCE}" --frames "${FRAMES[@]}" --methods "${METHOD}" \
  --dataset-root "${PCE_DATASET_ROOT}" --results-root "${PCE_RESULTS_ROOT}" \
  --run-name "${METHOD}/run_${SLURM_ARRAY_JOB_ID}" \
  --image-size 512 --tmp-dir "${TMPDIR}" --skip-pcqm
