#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"

TIME_WORKFLOW_NAME=dataset_diagnostics
TIME_TASK_NAME=window_audit_and_features
TIME_STATUS_NAME=dataset_diagnostics
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

time_workflow_init
if [ ! -d "$TIME_DATASET" ]; then
    echo "TIME dataset directory not found: $TIME_DATASET" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
diagnostics_root="$TIME_METADATA/window_audit"
config_path="$PROJECT_ROOT/src/timebench/config/datasets.yaml"

time_stage_start window_audit
time_task_start "audit configured TIME queries and model-effective contexts"
audit_command=(
    uv run --no-sync python
    "$PROJECT_ROOT/scripts/audit_time_windows.py"
    --config "$config_path"
    --output-dir "$diagnostics_root"
)
srun --ntasks=1 "${audit_command[@]}"
time_task_complete
time_stage_complete
source "$PROJECT_ROOT/src/slurm/export_dataset_metadata.sh"

time_stage_start dataset_features
time_task_start "reuse or compute full-series TIME statistics and STL features"
feature_command=(
    uv run --no-sync python -m timebench.feature.features_runner
    --all
    --config "$config_path"
    --input-format hf
    --split full
    --dataset_dir "$TIME_DATASET"
    --output_dir "$TIME_METADATA"
)
srun --ntasks=1 "${feature_command[@]}"
time_task_complete
time_stage_complete
source "$PROJECT_ROOT/src/slurm/export_dataset_metadata.sh"
time_workflow_complete
