#!/bin/bash
#SBATCH --job-name=pce_monitor
#SBATCH --account=mcsei7083
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/incomplete_task_monitor/%x_%j.out
#SBATCH --error=logs/incomplete_task_monitor/%x_%j.err

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs/incomplete_task_monitor

PYTHON=/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/python
INTERVAL_SECONDS=${PCE_MONITOR_INTERVAL_SECONDS:-10800}
MAX_CHECKS=${PCE_MONITOR_MAX_CHECKS:-40}

for CHECK_INDEX in $(seq 1 "$MAX_CHECKS"); do
  echo "[monitor] check ${CHECK_INDEX}/${MAX_CHECKS} $(date --iso-8601=seconds)"
  "$PYTHON" scripts/monitor_incomplete_snellius_tasks.py
  if [ "$CHECK_INDEX" -lt "$MAX_CHECKS" ]; then
    sleep "$INTERVAL_SECONDS"
  fi
done
