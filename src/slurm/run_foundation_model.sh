#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
source "$PROJECT_ROOT/src/slurm/foundation_model_runners.sh"

model="${TIME_MODEL:?TIME_MODEL must name one registered foundation model}"
runner=""
for model_index in "${!FOUNDATION_MODELS[@]}"; do
    if [ "${FOUNDATION_MODELS[$model_index]}" = "$model" ]; then
        runner="${FOUNDATION_RUNNERS[$model_index]}"
        break
    fi
done
if [ -z "$runner" ]; then
    echo "unknown TIME_MODEL=$model; expected one of: ${FOUNDATION_MODELS[*]}" >&2
    exit 2
fi

TIME_WORKFLOW_NAME=foundation_models
TIME_EXPERIMENT=foundation_models
TIME_TASK_NAME="$model"
TIME_STATUS_NAME="$model"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
TIME_RESULT_SCOPE="$TIME_OUTPUTS/foundation_models/tasks/$model"
export TIME_WORKFLOW_NAME TIME_EXPERIMENT TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID TIME_RESULT_SCOPE
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start evaluate
time_task_start "model=$model runner=$runner environment=uv covariate_mode=${TIME_COVARIATE_MODE:-none} target_mode=${TIME_TARGET_MODE:-auto}"
TIME_RUN_SCRIPT="$runner" source "$PROJECT_ROOT/src/slurm/run_time_script.sh"
time_task_complete
time_stage_complete
time_workflow_complete
