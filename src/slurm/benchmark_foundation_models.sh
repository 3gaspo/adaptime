#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
source "$PROJECT_ROOT/src/slurm/foundation_model_runners.sh"
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

TIME_WORKFLOW_NAME=foundation_models
TIME_TASK_NAME=all_foundation_models
TIME_STATUS_NAME=workflow
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID

model_indices=("${!FOUNDATION_MODELS[@]}")
if [ -n "${TIME_MODEL_INDICES:-}" ]; then
    read -r -a model_indices <<< "$TIME_MODEL_INDICES"
fi

time_workflow_init
time_stage_start evaluate
for model_index in "${model_indices[@]}"; do
    if ! [[ "$model_index" =~ ^[0-9]+$ ]] || [ "$model_index" -ge "$FOUNDATION_MODEL_COUNT" ]; then
        echo "TIME_MODEL_INDICES entries must be between 0 and $((FOUNDATION_MODEL_COUNT - 1))" >&2
        exit 2
    fi
    model="${FOUNDATION_MODELS[$model_index]}"
    runner="${FOUNDATION_RUNNERS[$model_index]}"
    time_task_start "model=$model runner=$runner"
    TIME_MODEL_INDEX="$model_index" TIME_LAUNCH_ID="$TIME_LAUNCH_ID" \
        bash "$PROJECT_ROOT/src/slurm/run_foundation_model.sh"
    time_task_complete
done
time_stage_complete

time_stage_start summarize
time_task_start "foundation_model_summary outputs=$TIME_OUTPUTS"
TIME_LAUNCH_ID="$TIME_LAUNCH_ID" \
    bash "$PROJECT_ROOT/src/slurm/summarize_foundation_models.sh"
time_task_complete
time_stage_complete
time_workflow_complete
