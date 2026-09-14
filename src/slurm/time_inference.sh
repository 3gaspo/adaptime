#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

dataset="${TIME_INFERENCE_DATASET:-${1:-SG_Weather/D}}"
term="${TIME_INFERENCE_TERM:-${2:-short}}"
samples="${TIME_INFERENCE_SAMPLES:-${3:-30}}"
if ! [[ "$samples" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAMPLES must be a positive integer, received: $samples" >&2
    exit 2
fi

read -r -a methods <<< "${TIME_INFERENCE_METHODS:-vanilla covariate_prediction bayes_covariate_prediction full_ridge_shared tsrag}"
if [ "${#methods[@]}" -eq 0 ]; then
    echo "TIME_INFERENCE_METHODS must contain at least one method" >&2
    exit 2
fi

TIME_WORKFLOW_NAME=time_inference
TIME_TASK_NAME="${dataset}_${term}"
TIME_STATUS_NAME="$TIME_TASK_NAME"
TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
artifact_root="${ADAPTIME_OUTPUT_ROOT:-$TIME_OUTPUTS/adaptime}"
output="${TIME_INFERENCE_OUTPUT:-$artifact_root/time_inference/$dataset/$term/${TIME_LAUNCH_ID}.json}"
export TIME_WORKFLOW_NAME TIME_TASK_NAME TIME_STATUS_NAME TIME_LAUNCH_ID
source "$PROJECT_ROOT/src/slurm/workflow_common.sh"

command=(
    uv run --no-sync python -m timebench.scripts.time_inference
    --artifact-root "$artifact_root"
    --dataset "$dataset"
    --term "$term"
    --methods "${methods[@]}"
    --samples "$samples"
    --seed "${TIME_INFERENCE_SEED:-1}"
    --model "${ADAPTIME_MODEL:-chronos2}"
    --device "${ADAPTIME_DEVICE:-cuda}"
    --output "$output"
    --tsrag-chronos-bolt-path "${TSRAG_CHRONOS_BOLT_PATH:-$TIME_WEIGHTS/chronos-bolt-base}"
    --tsrag-retriever-path "${TSRAG_RETRIEVER_PATH:-$TIME_WEIGHTS/chronos-t5-base}"
    --tsrag-checkpoint-path "${TSRAG_CHECKPOINT_PATH:-$TIME_WEIGHTS/ts-rag}"
)
[ -z "${ADAPTIME_MODEL_PATH:-}" ] || command+=(--model-path "$ADAPTIME_MODEL_PATH")
[ -z "${ADAPTIME_WEIGHTS_ID:-}" ] || command+=(--weights-id "$ADAPTIME_WEIGHTS_ID")
[ -z "${TIME_INFERENCE_PREPARED:-}" ] || command+=(--prepared "$TIME_INFERENCE_PREPARED")
[ -z "${TIME_INFERENCE_FIT_EXTRACTION:-}" ] || command+=(--fit-extraction "$TIME_INFERENCE_FIT_EXTRACTION")
[ -z "${TIME_INFERENCE_ADAPTATION_MODEL:-}" ] || command+=(--adaptation-model "$TIME_INFERENCE_ADAPTATION_MODEL")
[ -z "${TIME_INFERENCE_TSRAG_EXTRACTION:-}" ] || command+=(--tsrag-extraction "$TIME_INFERENCE_TSRAG_EXTRACTION")

time_workflow_init
time_stage_start benchmark
time_task_start "dataset=$dataset term=$term samples=$samples methods=${methods[*]} output=$output"
srun --ntasks=1 "${command[@]}"
time_task_complete
time_stage_complete
time_workflow_complete
