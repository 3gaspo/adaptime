#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set by the Slurm front}"
NNI_FILE="${TIME_NNI_FILE:-$HOME/codes/.secrets/nni}"
if [ ! -f "$NNI_FILE" ]; then
    echo "missing Selena NNI file: $NNI_FILE" >&2
    exit 1
fi

SELENA_NNI="$(sed -n '1p' "$NNI_FILE" | tr -d '[:space:]')"
selena_nni="${SELENA_NNI,,}"
if [[ ! "$selena_nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "the Selena NNI file must contain one valid account name" >&2
    exit 1
fi

PROJECT_NAME="$(basename "$PROJECT_ROOT")"
TIME_SCRATCH_ROOT="${TIME_SCRATCH_ROOT:-/scratch/users/$selena_nni/codes/$PROJECT_NAME}"
TIME_DATA_ROOT="${TIME_DATA_ROOT:-$TIME_SCRATCH_ROOT/datasets}"
TIME_DATASET="${TIME_DATASET:-$TIME_DATA_ROOT/hf_dataset}"
TIME_WEIGHTS="${TIME_WEIGHTS:-$TIME_SCRATCH_ROOT/weights}"
TIME_OUTPUTS="${TIME_OUTPUTS:-$TIME_SCRATCH_ROOT/outputs}"
TIME_LOGS="${TIME_LOGS:-$TIME_SCRATCH_ROOT/logs}"

export SELENA_NNI TIME_SCRATCH_ROOT TIME_DATA_ROOT TIME_DATASET TIME_WEIGHTS
export TIME_OUTPUTS TIME_LOGS
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$TIME_OUTPUTS" "$TIME_LOGS"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
