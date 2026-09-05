#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/src/slurm/runtime_paths.sh"
cd "$ROOT_DIR"

: "${TIME_COVARIATE_MODE:?TIME_COVARIATE_MODE must be set by the comparison workflow}"
: "${TIME_TARGET_MODE:?TIME_TARGET_MODE must be set by the comparison workflow}"
: "${TIME_TASKS_ROOT:?TIME_TASKS_ROOT must be set by the comparison workflow}"

python experiments/chronos2.py \
    --dataset all_multivariate_datasets \
    --output-dir "$TIME_TASKS_ROOT" \
    --covariate-mode "$TIME_COVARIATE_MODE" \
    --target-mode "$TIME_TARGET_MODE"
