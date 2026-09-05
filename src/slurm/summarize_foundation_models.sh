#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
source "$PROJECT_ROOT/src/slurm/foundation_model_runners.sh"

TIME_WORKFLOW_NAME=foundation_summary
TIME_TASK_NAME=macro_mase_and_timing
TIME_STATUS_NAME=summary
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start summarize
time_task_start "foundation_model_summary outputs=$TIME_OUTPUTS"

case "${TIME_COVARIATE_MODE:-none}" in
    none) experiment=expe_uni ;;
    future_included) experiment=expe_covar ;;
    *)
        echo "Unknown TIME_COVARIATE_MODE=${TIME_COVARIATE_MODE:-}" >&2
        exit 2
        ;;
esac

summary_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
    --results-dir "$TIME_OUTPUTS/results/$experiment"
    --models "${FOUNDATION_MODELS[@]}"
    --launch-id "$TIME_LAUNCH_ID"
    --status-dir "$TIME_LOGS/workflow_status/foundation_models/$TIME_LAUNCH_ID"
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
analysis_root="$TIME_OUTPUTS/results/$experiment/analysis/$TIME_LAUNCH_ID"
time_task_start "mase_vs_features features=$TIME_METADATA/stl_features output=$analysis_root"
plot_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/plot_feature_performance.py"
    --features-root "$TIME_METADATA/stl_features"
    --results-dir "$TIME_OUTPUTS/results/$experiment"
    --output "$analysis_root/mase_vs_features.svg"
    --models "${FOUNDATION_MODELS[@]}"
    --launch-id "$TIME_LAUNCH_ID"
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
