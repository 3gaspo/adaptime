#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$TIME_DATASET" ]; then
    echo "TIME dataset directory not found: $TIME_DATASET" >&2
    exit 1
fi

TIME_WORKFLOW_NAME=tsrag_comparison
TIME_TASK_NAME=tsrag_vs_vanilla_vs_full_ridge
TIME_STATUS_NAME=tsrag_comparison
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
RIDGE_OUTPUT_ROOT="${TSRAG_RIDGE_OUTPUT_ROOT:-${ADAPTIME_OUTPUT_ROOT:-$TIME_OUTPUTS/adaptime}}"
TSRAG_OUTPUT_ROOT="${TSRAG_OUTPUT_ROOT:-$TIME_OUTPUTS/tsrag_comparison}"
TIME_RESULT_SCOPE="$TSRAG_OUTPUT_ROOT/tasks/tsrag/univariate"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID TIME_RESULT_SCOPE
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

run_command() {
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 "$@"
    else
        "$@"
    fi
}

run_tsrag_and_table() {
    local -a command
    command=(
        uv run --no-sync python -m timebench.scripts.run_tsrag_comparison
        --datasets "${TSRAG_DATASETS:-${ADAPTIME_DATASETS:-all_datasets}}"
        --config "${ADAPTIME_DATASET_CONFIG:-$PROJECT_ROOT/src/timebench/config/datasets.yaml}"
        --output-root "$TSRAG_OUTPUT_ROOT"
        --ridge-output-root "$RIDGE_OUTPUT_ROOT"
        --device "${TSRAG_DEVICE:-cuda}"
        --model-batch-size "${TSRAG_MODEL_BATCH_SIZE:-256}"
        --arrow-cache-items "${TSRAG_ARROW_CACHE_ITEMS:-2}"
        --config-policy "${ADAPTIME_CONFIG_POLICY:-error}"
        --repeat-policy "${ADAPTIME_REPEAT_POLICY:-selected}"
        --chronos-bolt-path "${TSRAG_CHRONOS_BOLT_PATH:-$TIME_WEIGHTS/chronos-bolt-base}"
        --retriever-path "${TSRAG_RETRIEVER_PATH:-$TIME_WEIGHTS/chronos-t5-base}"
        --checkpoint-path "${TSRAG_CHECKPOINT_PATH:-$TIME_WEIGHTS/ts-rag}"
    )
    [ -z "${TSRAG_TERMS:-${ADAPTIME_TERMS:-}}" ] || command+=(--terms "${TSRAG_TERMS:-${ADAPTIME_TERMS:-}}")
    [ -z "${TSRAG_RIDGE_LAUNCH_ID:-}" ] || command+=(--ridge-launch-id "$TSRAG_RIDGE_LAUNCH_ID")
    run_command "${command[@]}"
}

time_workflow_init
time_stage_start tsrag_and_tables
time_task_start "pinned TS-RAG evaluation from main Adaptime ridge results and scaled-MASE/time tables"
run_tsrag_and_table
time_task_complete
time_stage_complete
time_workflow_complete
