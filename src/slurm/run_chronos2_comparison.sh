#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"

comparison="${TIME_COMPARISON:?TIME_COMPARISON must be multivariate, univariate, or covariate}"
case "$comparison" in
    multivariate)
        TIME_COVARIATE_MODE=none
        TIME_TARGET_MODE=multivariate
        experiment=expe_uni
        aggregate=chronos2_multivariate
        ;;
    univariate)
        TIME_COVARIATE_MODE=none
        TIME_TARGET_MODE=univariate
        experiment=expe_uni
        aggregate=chronos2_univariate
        ;;
    covariate)
        TIME_COVARIATE_MODE=past_targets
        TIME_TARGET_MODE=univariate
        experiment=expe_covar
        aggregate=chronos2_past_targets
        ;;
    *)
        echo "Unknown TIME_COMPARISON=$comparison" >&2
        exit 2
        ;;
esac
export TIME_COVARIATE_MODE TIME_TARGET_MODE

TIME_WORKFLOW_NAME=chronos2_comparison
TIME_TASK_NAME="$comparison"
TIME_STATUS_NAME="$comparison"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start evaluate
time_task_start "chronos2 comparison=$comparison covariate_mode=$TIME_COVARIATE_MODE target_mode=$TIME_TARGET_MODE"
TIME_RUN_SCRIPT=run_chronos2_comparison.sh source "$PROJECT_ROOT/src/slurm/run_time_script.sh"
time_task_complete
time_stage_complete

time_stage_start summarize
aggregate_dir="$TIME_OUTPUTS/results/$experiment/aggregates/$aggregate/$TIME_LAUNCH_ID"
mkdir -p "$aggregate_dir"
summary_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
    --results-dir "$TIME_OUTPUTS/results/$experiment"
    --models chronos2
    --launch-id "$TIME_LAUNCH_ID"
    --target-mode "$TIME_TARGET_MODE"
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
