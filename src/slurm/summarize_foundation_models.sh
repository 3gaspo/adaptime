#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"

TIME_WORKFLOW_NAME=foundation_summary
TIME_TASK_NAME=macro_mase_and_timing
TIME_STATUS_NAME=summary
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
time_stage_start summarize
time_task_start "foundation_model_summary outputs=$TIME_OUTPUTS"

summary_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
)

if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun --ntasks=1 "${summary_command[@]}"
else
    "${summary_command[@]}"
fi

time_task_complete
time_stage_complete
time_workflow_complete
