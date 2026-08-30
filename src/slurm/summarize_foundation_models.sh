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

if [ -n "${SUMMARY_PYTHON:-}" ]; then
    "$SUMMARY_PYTHON" "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda run -n "${TIME_SUMMARY_ENV:-time_chronos2}" \
        python "$PROJECT_ROOT/scripts/compute_foundation_summary.py"
fi

time_task_complete
time_stage_complete
time_workflow_complete
