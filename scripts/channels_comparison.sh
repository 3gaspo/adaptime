#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/channels_comparison.sh dgx|selena" >&2
}

cluster="${1:-}"
case "$cluster" in
    dgx|selena) ;;
    *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$cluster" = selena ]; then
    export PROJECT_ROOT
    source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
else
    TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
    export TIME_STORAGE_ROOT
    source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
fi
mkdir -p "$TIME_LOGS"

launch_id="${TIME_LAUNCH_ID:-${cluster}_channels_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
comparisons=(multivariate univariate covariate)

for comparison in "${comparisons[@]}"; do
    if [ "$cluster" = selena ]; then
        front="$PROJECT_ROOT/slurm/selena/chronos2_comparison/${comparison}_selena.slurm"
    else
        front="$PROJECT_ROOT/slurm/dgx/chronos2_comparison/${comparison}.slurm"
    fi
    job_id="$(
        sbatch --parsable \
            --export="ALL,TIME_LAUNCH_ID=$launch_id" \
            "$front"
    )"
    job_id="${job_id%%;*}"
    echo "channels comparison submitted mode=$comparison job_id=$job_id launch_id=$launch_id"
done
