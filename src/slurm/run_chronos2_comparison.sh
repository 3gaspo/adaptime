#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"

comparison="${TIME_COMPARISON:?TIME_COMPARISON must be multivariate, univariate, or covariate}"
TIME_REUSE_FROM="${TIME_REUSE_FROM:-}"
TIME_REUSE_IF_AVAILABLE_FROM=""
case "$comparison" in
    multivariate)
        TIME_COVARIATE_MODE=none
        TIME_TARGET_MODE=multivariate
        TIME_REUSE_IF_AVAILABLE_FROM="${TIME_REUSE_MULTIVARIATE_FROM:-$TIME_OUTPUTS/foundation_models/tasks/chronos2/multivariate}"
        ;;
    univariate)
        TIME_COVARIATE_MODE=none
        TIME_TARGET_MODE=univariate
        TIME_REUSE_FROM="${TIME_REUSE_UNIVARIATE_FROM:-}"
        ;;
    covariate)
        TIME_COVARIATE_MODE=past_targets
        TIME_TARGET_MODE=univariate
        TIME_REUSE_FROM="${TIME_REUSE_COVARIATE_FROM:-}"
        ;;
    *)
        echo "Unknown TIME_COMPARISON=$comparison" >&2
        exit 2
        ;;
esac
TIME_EXPERIMENT=channels_comparison
TIME_TASKS_ROOT="$TIME_OUTPUTS/channels_comparison/tasks/$comparison"
export TIME_COVARIATE_MODE TIME_TARGET_MODE TIME_REUSE_FROM TIME_REUSE_IF_AVAILABLE_FROM TIME_EXPERIMENT TIME_TASKS_ROOT

TIME_WORKFLOW_NAME=channels_comparison
TIME_TASK_NAME="$comparison"
TIME_STATUS_NAME="$comparison"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
TIME_RESULT_SCOPE="$TIME_TASKS_ROOT/chronos2/$TIME_TARGET_MODE"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID TIME_RESULT_SCOPE
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start evaluate
time_task_start "chronos2 comparison=$comparison covariate_mode=$TIME_COVARIATE_MODE target_mode=$TIME_TARGET_MODE"
TIME_RUN_SCRIPT=run_chronos2_comparison.sh source "$PROJECT_ROOT/src/slurm/run_time_script.sh"
time_task_complete
time_stage_complete

time_stage_start summarize
aggregate_dir="$TIME_OUTPUTS/channels_comparison/summary/$TIME_LAUNCH_ID/$comparison"
mkdir -p "$aggregate_dir"
summary_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
    --results-dir "$TIME_TASKS_ROOT"
    --models chronos2
    --model-status chronos2=completed,0
    --launch-id "$TIME_LAUNCH_ID"
    --target-mode "$TIME_TARGET_MODE"
    --config-policy "${TIME_CONFIG_POLICY:-error}"
    --repeat-policy "${TIME_REPEAT_POLICY:-selected}"
    --csv "$aggregate_dir/foundation_model_summary.csv"
    --markdown "$aggregate_dir/foundation_model_summary.md"
)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${summary_command[@]}"
else
    "${summary_command[@]}"
fi
time_stage_complete
time_workflow_complete
