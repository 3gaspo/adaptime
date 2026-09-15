#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/submit_seasonal_naive.sh dgx|selena [shared|project]" >&2
}

cluster="${1:-}"
case "$cluster" in
    dgx|selena) ;;
    *) usage; exit 2 ;;
esac

seasonal_scope="${2:-${TIME_SEASONAL_SCOPE:-shared}}"
case "$seasonal_scope" in
    shared|project) ;;
    *) usage; exit 2 ;;
esac
export TIME_SEASONAL_SCOPE="$seasonal_scope"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$cluster" = selena ]; then
    export PROJECT_ROOT
    source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
    front="$PROJECT_ROOT/slurm/selena/foundation_models/seasonal_naive_selena.slurm"
else
    TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
    export TIME_STORAGE_ROOT
    source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
    front="$PROJECT_ROOT/slurm/dgx/foundation_models/seasonal_naive.slurm"
fi
mkdir -p "$TIME_LOGS"

launch_id="${TIME_LAUNCH_ID:-${cluster}_seasonal_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
job_id="$(
    sbatch --parsable \
        --export="ALL,TIME_LAUNCH_ID=$launch_id,OUTPUTS_ROOT=$TIME_SEASONAL_ROOT" \
        "$front"
)"
job_id="${job_id%%;*}"

echo "$seasonal_scope Seasonal Naive submitted job_id=$job_id launch_id=$launch_id"
echo "Seasonal task root: $TIME_SEASONAL_TASKS_ROOT"
echo "status: bash scripts/foundation_model_status.sh $cluster $launch_id"

