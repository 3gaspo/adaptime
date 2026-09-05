#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$TIME_DATASET" ]; then
    echo "TIME dataset directory not found: $TIME_DATASET" >&2
    exit 1
fi

TIME_WORKFLOW_NAME=adaptime_comparison
TIME_TASK_NAME="${ADAPTIME_MODEL:-chronos2}_${ADAPTIME_TARGET_MODE:-univariate}"
TIME_STATUS_NAME="$TIME_TASK_NAME"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
ADAPTIME_OUTPUT_ROOT_VALUE="${ADAPTIME_OUTPUT_ROOT:-$TIME_OUTPUTS/adaptime}"
TIME_RESULT_SCOPE="$ADAPTIME_OUTPUT_ROOT_VALUE/tasks/${ADAPTIME_MODEL:-chronos2}/${ADAPTIME_TARGET_MODE:-univariate}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID TIME_RESULT_SCOPE
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

run_adaptime() {
    local -a k_values alpha_values command
    read -r -a k_values <<< "${ADAPTIME_K_VALUES:-1 5 10 15}"
    read -r -a alpha_values <<< "${ADAPTIME_ALPHA_VALUES:-0.001 0.01 0.1}"
    command=(
        uv run --no-sync python -m timebench.scripts.run_adaptation_stage
        --stage run
        --datasets "${ADAPTIME_DATASETS:-all_datasets}"
        --config "${ADAPTIME_DATASET_CONFIG:-$PROJECT_ROOT/src/timebench/config/datasets.yaml}"
        --output-root "$ADAPTIME_OUTPUT_ROOT_VALUE"
        --model "${ADAPTIME_MODEL:-chronos2}"
        --device "${ADAPTIME_DEVICE:-cuda}"
        --target-mode "${ADAPTIME_TARGET_MODE:-univariate}"
        --max-context-length "${ADAPTIME_MAX_CONTEXT_LENGTH:-2048}"
        --datastore-stride-multiple "${ADAPTIME_DATASTORE_STRIDE_MULTIPLE:-1}"
        --representation "${ADAPTIME_REPRESENTATION:-instance}"
        --distance-metric "${ADAPTIME_DISTANCE_METRIC:-euclidean}"
        --retrieval-scope "${ADAPTIME_RETRIEVAL_SCOPE:-all}"
        --minimum-overlap-fraction "${ADAPTIME_MINIMUM_OVERLAP_FRACTION:-0.8}"
        --max-k "${ADAPTIME_MAX_K:-15}"
        --k "${k_values[@]}"
        --alpha "${alpha_values[@]}"
        --model-batch-size "${ADAPTIME_MODEL_BATCH_SIZE:-64}"
        --query-block-size "${ADAPTIME_QUERY_BLOCK_SIZE:-256}"
        --datastore-block-size "${ADAPTIME_DATASTORE_BLOCK_SIZE:-4096}"
        --arrow-cache-items "${ADAPTIME_ARROW_CACHE_ITEMS:-2}"
        --ridge-chunk-size "${ADAPTIME_RIDGE_CHUNK_SIZE:-1024}"
        --seed "${ADAPTIME_SEED:-1}"
        --config-policy "${ADAPTIME_CONFIG_POLICY:-error}"
        --repeat-policy "${ADAPTIME_REPEAT_POLICY:-selected}"
    )
    [ -z "${ADAPTIME_TERMS:-}" ] || command+=(--terms "$ADAPTIME_TERMS")
    [ -z "${ADAPTIME_MODEL_PATH:-}" ] || command+=(--model-path "$ADAPTIME_MODEL_PATH")
    [ -z "${ADAPTIME_WEIGHTS_ID:-}" ] || command+=(--weights-id "$ADAPTIME_WEIGHTS_ID")
    [ -z "${ADAPTIME_TRAIN_LENGTH:-}" ] || command+=(--adaptation-train-length "$ADAPTIME_TRAIN_LENGTH")
    [ -z "${ADAPTIME_VALIDATION_LENGTH:-}" ] || command+=(--adaptation-validation-length "$ADAPTIME_VALIDATION_LENGTH")
    [ -z "${ADAPTIME_ADAPTATION_STRIDE:-}" ] || command+=(--adaptation-stride "$ADAPTIME_ADAPTATION_STRIDE")
    [ -z "${ADAPTIME_RETRIEVAL_PERIOD:-}" ] || command+=(--retrieval-period "$ADAPTIME_RETRIEVAL_PERIOD")
    [ -z "${ADAPTIME_DATASTORE_LENGTH:-}" ] || command+=(--datastore-length "$ADAPTIME_DATASTORE_LENGTH")
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 "${command[@]}"
    else
        "${command[@]}"
    fi
}

time_workflow_init
time_stage_start run
time_task_start "TIME-wide Adaptime model=${ADAPTIME_MODEL:-chronos2} target_mode=${ADAPTIME_TARGET_MODE:-univariate}"
run_adaptime
time_task_complete
time_stage_complete
time_workflow_complete
