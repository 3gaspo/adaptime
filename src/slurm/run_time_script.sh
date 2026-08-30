#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
TIME_RUN_SCRIPT="${TIME_RUN_SCRIPT:?Set TIME_RUN_SCRIPT to an existing run_*.sh filename}"

case "$TIME_RUN_SCRIPT" in
    */*|*\\*)
        echo "TIME_RUN_SCRIPT must not contain a directory" >&2
        exit 2
        ;;
    run_*.sh) ;;
    *)
        echo "TIME_RUN_SCRIPT must be a run_*.sh filename without directories" >&2
        exit 2
        ;;
esac

run_path="$PROJECT_ROOT/scripts/$TIME_RUN_SCRIPT"
if [ ! -f "$run_path" ]; then
    echo "TIME runner not found: $run_path" >&2
    exit 2
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TIME runner: $TIME_RUN_SCRIPT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] datasets: $TIME_DATASET"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] weights: $TIME_WEIGHTS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] outputs: $TIME_OUTPUTS"

bash "$run_path"
