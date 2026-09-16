#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
# Adaptime jobs own these roots; inherited settings from another checkout
# must never redirect the baseline or Ridge artifacts into that project.
if [ -n "${SELENA_NNI:-}" ]; then
    ADAPTIME_PROJECT_ROOT="$TIME_STORAGE_ROOT/codes/adaptime"
else
    ADAPTIME_PROJECT_ROOT="$PROJECT_ROOT"
fi
export TIME_SCRATCH_ROOT="$ADAPTIME_PROJECT_ROOT"
export OUTPUTS_ROOT="$ADAPTIME_PROJECT_ROOT/outputs"
export LOGS_ROOT="$ADAPTIME_PROJECT_ROOT/logs"
export TIME_OUTPUTS="$OUTPUTS_ROOT" TIME_LOGS="$LOGS_ROOT"
export ADAPTIME_OUTPUT_ROOT="$OUTPUTS_ROOT/adaptime"
export TIME_PROJECT_NAME=adaptime
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$TIME_DATASET" ]; then
    echo "TIME dataset directory not found: $TIME_DATASET" >&2
    exit 1
fi

ADAPTIME_METHOD_VALUE="${ADAPTIME_METHOD:-ridge}"
ADAPTIME_STAGE_VALUE="${ADAPTIME_STAGE:-all}"
TIME_WORKFLOW_NAME=adaptime
TIME_TASK_NAME="${ADAPTIME_METHOD_VALUE}_${ADAPTIME_STAGE_VALUE}"
TIME_STATUS_NAME="$TIME_TASK_NAME"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
ADAPTIME_OUTPUT_ROOT_VALUE="${ADAPTIME_OUTPUT_ROOT:-$TIME_OUTPUTS/adaptime}"
ADAPTIME_EXCLUDE_DATASETS_VALUE="${ADAPTIME_EXCLUDE_DATASETS:-Coastal_T_S/5T,current_velocity/20T,azure2019_D/5T,azure2019_I/5T}"
TIME_RESULT_SCOPE="$ADAPTIME_OUTPUT_ROOT_VALUE"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID TIME_RESULT_SCOPE
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

run_adaptime() {
    local -a k_values alpha_values fitting_scopes command
    read -r -a k_values <<< "${ADAPTIME_K_VALUES:-1 5 10 15}"
    read -r -a alpha_values <<< "${ADAPTIME_ALPHA_VALUES:-0.001 0.01 0.1}"
    read -r -a fitting_scopes <<< "${ADAPTIME_FITTING_SCOPES:-all same_series}"
    command=(
        uv run --no-sync python -m timebench.scripts.run_adaptation_stage
        --stage "$ADAPTIME_STAGE_VALUE"
        --method "$ADAPTIME_METHOD_VALUE"
        --datasets "${ADAPTIME_DATASETS:-all_datasets}"
        --exclude-datasets "$ADAPTIME_EXCLUDE_DATASETS_VALUE"
        --config "${ADAPTIME_DATASET_CONFIG:-$PROJECT_ROOT/src/timebench/config/datasets.yaml}"
        --output-root "$ADAPTIME_OUTPUT_ROOT_VALUE"
        --seasonal-results-path "${ADAPTIME_SEASONAL_RESULTS_PATH:-$TIME_SEASONAL_TASKS_ROOT/seasonal_naive}"
        --model "${ADAPTIME_MODEL:-chronos2}"
        --device "${ADAPTIME_DEVICE:-cuda}"
        --target-mode "${ADAPTIME_TARGET_MODE:-univariate}"
        --datastore-stride-multiple "${ADAPTIME_DATASTORE_STRIDE_MULTIPLE:-1}"
        --representation "${ADAPTIME_REPRESENTATION:-instance}"
        --distance-metric "${ADAPTIME_DISTANCE_METRIC:-euclidean}"
        --retrieval-scope "${ADAPTIME_RETRIEVAL_SCOPE:-all}"
        --minimum-overlap-fraction "${ADAPTIME_MINIMUM_OVERLAP_FRACTION:-0.8}"
        --minimum-query-finite-fraction "${ADAPTIME_MINIMUM_QUERY_FINITE_FRACTION:-0.8}"
        --max-k "${ADAPTIME_MAX_K:-15}"
        --k "${k_values[@]}"
        --alpha "${alpha_values[@]}"
        --model-batch-size "${ADAPTIME_MODEL_BATCH_SIZE:-64}"
        --query-block-size "${ADAPTIME_QUERY_BLOCK_SIZE:-256}"
        --datastore-block-size "${ADAPTIME_DATASTORE_BLOCK_SIZE:-4096}"
        --arrow-cache-items "${ADAPTIME_ARROW_CACHE_ITEMS:-2}"
        --ridge-chunk-size "${ADAPTIME_RIDGE_CHUNK_SIZE:-1024}"
        --bootstrap-replications "${ADAPTIME_BOOTSTRAP_REPLICATIONS:-1000}"
        --fitting-scope "${fitting_scopes[@]}"
        --rolling-k "${ADAPTIME_ROLLING_K:-15}"
        --rolling-alpha "${ADAPTIME_ROLLING_ALPHA:-1.0}"
        --rolling-n-fitting-dates "${ADAPTIME_ROLLING_N_FITTING_DATES:-100}"
        --rolling-minimum-fitting-dates "${ADAPTIME_ROLLING_MINIMUM_FITTING_DATES:-64}"
        --rolling-fitting-stride-multiple "${ADAPTIME_ROLLING_FITTING_STRIDE_MULTIPLE:-1}"
        --rolling-max-datastore-windows "${ADAPTIME_ROLLING_MAX_DATASTORE_WINDOWS:-10000}"
        --rolling-datastore-stride-multiple "${ADAPTIME_ROLLING_DATASTORE_STRIDE_MULTIPLE:-1}"
        --rolling-datastore-block-size "${ADAPTIME_ROLLING_DATASTORE_BLOCK_SIZE:-512}"
        --seed "${ADAPTIME_SEED:-1}"
        --tsrag-model-batch-size "${TSRAG_MODEL_BATCH_SIZE:-256}"
        --config-policy "${ADAPTIME_CONFIG_POLICY:-error}"
        --repeat-policy "${ADAPTIME_REPEAT_POLICY:-selected}"
    )
    [ -z "${ADAPTIME_TERMS:-}" ] || command+=(--terms "$ADAPTIME_TERMS")
    [ -z "${ADAPTIME_MODEL_PATH:-}" ] || command+=(--model-path "$ADAPTIME_MODEL_PATH")
    [ -z "${ADAPTIME_WEIGHTS_ID:-}" ] || command+=(--weights-id "$ADAPTIME_WEIGHTS_ID")
    [ -z "${ADAPTIME_ADAPTATION_STRIDE:-}" ] || command+=(--adaptation-stride "$ADAPTIME_ADAPTATION_STRIDE")
    [ -z "${ADAPTIME_RETRIEVAL_PERIOD:-}" ] || command+=(--retrieval-period "$ADAPTIME_RETRIEVAL_PERIOD")
    [ -z "${ADAPTIME_RETRIEVAL_CONTEXT_LENGTH:-}" ] || command+=(--retrieval-context-length "$ADAPTIME_RETRIEVAL_CONTEXT_LENGTH")
    [ -z "${ADAPTIME_BOOTSTRAP_BLOCK_LENGTH:-}" ] || command+=(--bootstrap-block-length "$ADAPTIME_BOOTSTRAP_BLOCK_LENGTH")
    [ -z "${ADAPTIME_MAX_DATASTORE_WINDOWS:-}" ] || command+=(--max-datastore-windows "$ADAPTIME_MAX_DATASTORE_WINDOWS")
    [ -z "${ADAPTIME_MAX_FITTING_WINDOWS:-}" ] || command+=(--max-fitting-windows "$ADAPTIME_MAX_FITTING_WINDOWS")
    [ -z "${TSRAG_CHRONOS_BOLT_PATH:-}" ] || command+=(--tsrag-chronos-bolt-path "$TSRAG_CHRONOS_BOLT_PATH")
    [ -z "${TSRAG_RETRIEVER_PATH:-}" ] || command+=(--tsrag-retriever-path "$TSRAG_RETRIEVER_PATH")
    [ -z "${TSRAG_CHECKPOINT_PATH:-}" ] || command+=(--tsrag-checkpoint-path "$TSRAG_CHECKPOINT_PATH")
    if [ "$ADAPTIME_STAGE_VALUE" = report ] && [ "$ADAPTIME_METHOD_VALUE" = tsrag ] && [ -n "${ADAPTIME_RIDGE_RESULTS_PATH:-}" ]; then
        command+=(--ridge-results-path "$ADAPTIME_RIDGE_RESULTS_PATH")
    fi
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks=1 "${command[@]}"
    else
        "${command[@]}"
    fi
}

time_workflow_init
time_stage_start "$ADAPTIME_STAGE_VALUE"
time_task_start "TIME-wide Adaptime method=$ADAPTIME_METHOD_VALUE stage=$ADAPTIME_STAGE_VALUE"
run_adaptime
time_task_complete
time_stage_complete
time_workflow_complete
