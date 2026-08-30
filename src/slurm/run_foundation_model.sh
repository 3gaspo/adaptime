#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
source "$PROJECT_ROOT/src/slurm/foundation_model_runners.sh"

model_index="${TIME_MODEL_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if ! [[ "$model_index" =~ ^[0-9]+$ ]] || [ "$model_index" -ge "$FOUNDATION_MODEL_COUNT" ]; then
    echo "TIME_MODEL_INDEX or SLURM_ARRAY_TASK_ID must be between 0 and $((FOUNDATION_MODEL_COUNT - 1))" >&2
    exit 2
fi

model="${FOUNDATION_MODELS[$model_index]}"
runner="${FOUNDATION_RUNNERS[$model_index]}"
if [ "$model" = timesfm2p5 ] || [ "$model" = timesfm3 ]; then
    export TIMESFM_DIR="${TIMESFM_DIR:-$PROJECT_ROOT/experiments/timesfm_${model}}"
fi

TIME_WORKFLOW_NAME=foundation_models
TIME_TASK_NAME="$model"
TIME_STATUS_NAME="$(printf '%02d_%s' "$model_index" "$model")"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_ARRAY_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start evaluate
time_task_start "model=$model runner=$runner environment=uv"
TIME_RUN_SCRIPT="$runner" source "$PROJECT_ROOT/src/slurm/run_time_script.sh"
time_task_complete
time_stage_complete
time_workflow_complete
