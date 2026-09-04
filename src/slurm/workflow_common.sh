#!/bin/bash

set -euo pipefail

TIME_ACTIVE_STAGE=""
TIME_ACTIVE_TASK=""
TIME_WORKFLOW_DONE=false
TIME_LAUNCHED_AT=""

time_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

time_log() {
    echo "$(time_timestamp) | $*"
}

time_write_status() {
    local state="$1"
    local exit_code="$2"
    local temporary="${TIME_STATUS_FILE}.tmp.$$"
    {
        echo "workflow=$TIME_WORKFLOW_NAME"
        echo "launch_id=$TIME_LAUNCH_ID"
        echo "task=$TIME_TASK_NAME"
        echo "state=$state"
        echo "stage=${TIME_ACTIVE_STAGE:-none}"
        echo "active_task=${TIME_ACTIVE_TASK:-none}"
        echo "launched_at=$TIME_LAUNCHED_AT"
        echo "updated_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "exit_code=$exit_code"
        echo "slurm_job_id=${SLURM_JOB_ID:-none}"
        echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
        echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
    } > "$temporary"
    mv "$temporary" "$TIME_STATUS_FILE"
}

time_workflow_on_exit() {
    local status=$?
    trap - EXIT
    if [ "$TIME_WORKFLOW_DONE" != true ]; then
        if [ -n "$TIME_ACTIVE_TASK" ]; then
            time_log "task $TIME_ACTIVE_TASK completed status=failed exit_code=$status" >&2
        fi
        if [ -n "$TIME_ACTIVE_STAGE" ]; then
            time_log "stage $TIME_ACTIVE_STAGE completed status=failed exit_code=$status" >&2
        fi
        time_write_status failed "$status"
        time_log "workflow $TIME_WORKFLOW_NAME completed status=failed exit_code=$status" >&2
    fi
    exit "$status"
}

time_workflow_init() {
    : "${TIME_LOGS:?TIME_LOGS must be configured before workflow initialization}"
    : "${TIME_WORKFLOW_NAME:?TIME_WORKFLOW_NAME must be set}"
    : "${TIME_TASK_NAME:?TIME_TASK_NAME must be set}"
    : "${TIME_LAUNCH_ID:?TIME_LAUNCH_ID must be set}"
    local status_name="${TIME_STATUS_NAME:-$TIME_TASK_NAME}"
    status_name="${status_name//[^a-zA-Z0-9_.-]/_}"
    TIME_STATUS_ROOT="$TIME_LOGS/workflow_status/$TIME_WORKFLOW_NAME/$TIME_LAUNCH_ID"
    TIME_STATUS_FILE="$TIME_STATUS_ROOT/$status_name.status"
    TIME_LAUNCHED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    mkdir -p "$TIME_STATUS_ROOT"
    trap time_workflow_on_exit EXIT
    time_write_status running 0
    time_log "workflow $TIME_WORKFLOW_NAME started launch_id=$TIME_LAUNCH_ID task=$TIME_TASK_NAME slurm_job_id=${SLURM_JOB_ID:-none} slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-none} slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
}

time_stage_start() {
    TIME_ACTIVE_STAGE="$1"
    time_write_status running 0
    time_log "stage $TIME_ACTIVE_STAGE started"
}

time_stage_complete() {
    time_log "stage $TIME_ACTIVE_STAGE completed status=success"
    TIME_ACTIVE_STAGE=""
    time_write_status running 0
}

time_task_start() {
    TIME_ACTIVE_TASK="$*"
    time_write_status running 0
    time_log "task $TIME_ACTIVE_TASK started"
}

time_task_complete() {
    time_log "task $TIME_ACTIVE_TASK completed status=success"
    TIME_ACTIVE_TASK=""
    time_write_status running 0
}

time_workflow_complete() {
    TIME_ACTIVE_STAGE=""
    TIME_ACTIVE_TASK=""
    time_write_status completed 0
    TIME_WORKFLOW_DONE=true
    time_log "workflow $TIME_WORKFLOW_NAME completed status=success exit_code=0"
}
