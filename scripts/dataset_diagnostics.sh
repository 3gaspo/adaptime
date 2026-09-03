#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/dataset_diagnostics.sh dgx|selena" >&2
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
    front="$PROJECT_ROOT/slurm/selena/dataset_diagnostics_selena.slurm"
else
    TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
    export TIME_STORAGE_ROOT
    source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
    front="$PROJECT_ROOT/slurm/dgx/dataset_diagnostics.slurm"
fi
mkdir -p "$TIME_LOGS"

launch_id="${TIME_LAUNCH_ID:-${cluster}_dataset_diagnostics_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
job_id="$(
    sbatch --parsable \
        --export="ALL,TIME_LAUNCH_ID=$launch_id" \
        "$front"
)"
job_id="${job_id%%;*}"

echo "dataset diagnostics submitted job_id=$job_id launch_id=$launch_id"
echo "artifacts: $TIME_OUTPUTS/dataset_diagnostics/$launch_id"
echo "use --size detailed when synchronizing or publishing exact anomaly positions"
