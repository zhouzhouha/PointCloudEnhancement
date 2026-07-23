#!/bin/bash
set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement
mkdir -p logs/incomplete_task_monitor

PYTHON=/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/python
INTERVAL_SECONDS=${PCE_MONITOR_INTERVAL_SECONDS:-10800}
MAX_CHECKS=${PCE_MONITOR_MAX_CHECKS:-40}
LOCK=/tmp/pce_incomplete_monitor.lock

for CHECK_INDEX in $(seq 1 "$MAX_CHECKS"); do
  echo "[monitor-loop] check ${CHECK_INDEX}/${MAX_CHECKS} $(date --iso-8601=seconds)"
  /usr/bin/flock -n "$LOCK" "$PYTHON" scripts/monitor_incomplete_snellius_tasks.py
  if [ "$CHECK_INDEX" -lt "$MAX_CHECKS" ]; then
    sleep "$INTERVAL_SECONDS"
  fi
done
