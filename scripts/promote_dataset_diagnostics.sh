#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "usage: bash scripts/promote_dataset_diagnostics.sh JOB_ID" >&2
    exit 2
fi
JOB_ID="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PROJECT_ROOT
source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"

status_root="$TIME_LOGS/workflow_status/dataset_diagnostics"
mapfile -d '' status_files < <(
    find "$status_root" -type f -name dataset_diagnostics.status \
        -exec grep -lZ -x "slurm_job_id=$JOB_ID" {} +
)
if [ "${#status_files[@]}" -ne 1 ]; then
    echo "expected one diagnostics status for job $JOB_ID, found ${#status_files[@]}" >&2
    exit 1
fi
status_file="${status_files[0]}"
state="$(sed -n 's/^state=//p' "$status_file")"
if [ "$state" != completed ]; then
    echo "job $JOB_ID is not complete according to $status_file: state=$state" >&2
    exit 1
fi
launch_id="$(sed -n 's/^launch_id=//p' "$status_file")"

uv run --no-sync python "$PROJECT_ROOT/scripts/promote_dataset_diagnostics.py" \
    --source-audit "$TIME_OUTPUTS/dataset_diagnostics/$launch_id" \
    --source-features "$TIME_OUTPUTS/stl_features" \
    --metadata-root "$TIME_METADATA" \
    --log-export-dir "$TIME_LOGS/dataset_metadata/$JOB_ID" \
    --job-id "$JOB_ID"
