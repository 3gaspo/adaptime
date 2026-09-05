#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
source "$PROJECT_ROOT/src/slurm/foundation_model_runners.sh"

TIME_WORKFLOW_NAME=foundation_summary
TIME_EXPERIMENT=foundation_models
TIME_TASK_NAME=macro_mase_and_timing
TIME_STATUS_NAME=summary
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_EXPERIMENT TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start summarize
time_task_start "foundation_model_summary outputs=$TIME_OUTPUTS"

tasks_root="$TIME_OUTPUTS/foundation_models/tasks"
summary_root="$TIME_OUTPUTS/foundation_models/summary/$TIME_LAUNCH_ID"

summary_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
    --results-dir "$tasks_root"
    --models "${FOUNDATION_MODELS[@]}"
    --launch-id "$TIME_LAUNCH_ID"
    --status-dir "$TIME_LOGS/workflow_status/foundation_models/$TIME_LAUNCH_ID"
    --config-policy "${TIME_CONFIG_POLICY:-error}"
    --repeat-policy "${TIME_REPEAT_POLICY:-selected}"
    --csv "$summary_root/foundation_model_summary.csv"
    --markdown "$summary_root/foundation_model_summary.md"
)

if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${summary_command[@]}"
else
    "${summary_command[@]}"
fi

time_task_complete
time_stage_complete

status_root="$TIME_LOGS/workflow_status/foundation_models/$TIME_LAUNCH_ID"
incomplete_models=()
for model in "${FOUNDATION_MODELS[@]}"; do
    status_file="$status_root/$model.status"
    state=""
    exit_code=""
    if [ -f "$status_file" ]; then
        state="$(sed -n 's/^state=//p' "$status_file")"
        exit_code="$(sed -n 's/^exit_code=//p' "$status_file")"
    fi
    if [ "$state" != completed ] || [ "$exit_code" != 0 ]; then
        incomplete_models+=("$model:${state:-missing}:${exit_code:-unknown}")
    fi
done
if [ "${#incomplete_models[@]}" -gt 0 ]; then
    echo "Feature plot requires four successful model jobs; incomplete: ${incomplete_models[*]}" >&2
    exit 1
fi

time_stage_start feature_plot
analysis_root="$TIME_OUTPUTS/foundation_models/feature_analysis/$TIME_LAUNCH_ID"
time_task_start "mase_vs_features features=$TIME_METADATA/stl_features output=$analysis_root"
plot_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/plot_feature_performance.py"
    --features-root "$TIME_METADATA/stl_features"
    --results-dir "$tasks_root"
    --output "$analysis_root/mase_vs_features.svg"
    --models "${FOUNDATION_MODELS[@]}"
    --launch-id "$TIME_LAUNCH_ID"
    --config-policy "${TIME_CONFIG_POLICY:-error}"
    --repeat-policy "${TIME_REPEAT_POLICY:-selected}"
    --top 5
)

if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${plot_command[@]}"
else
    "${plot_command[@]}"
fi

time_task_complete
time_stage_complete
time_workflow_complete
