#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/submit_foundation_models.sh dgx|selena" >&2
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

source "$PROJECT_ROOT/src/slurm/foundation_model_runners.sh"
launch_id="${TIME_LAUNCH_ID:-${cluster}_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
model_jobs=()
for model in "${FOUNDATION_MODELS[@]}"; do
    if [ "$cluster" = selena ]; then
        front="$PROJECT_ROOT/slurm/selena/foundation_models/${model}_selena.slurm"
    else
        front="$PROJECT_ROOT/slurm/dgx/foundation_models/${model}.slurm"
    fi
    job_id="$(sbatch --parsable --export="ALL,TIME_LAUNCH_ID=$launch_id" "$front")"
    job_id="${job_id%%;*}"
    model_jobs+=("$job_id")
    echo "foundation model submitted model=$model job_id=$job_id launch_id=$launch_id"
done

dependency="$(IFS=:; echo "${model_jobs[*]}")"
if [ "$cluster" = selena ]; then
    summary_front="$PROJECT_ROOT/slurm/selena/foundation_summary_selena.slurm"
else
    summary_front="$PROJECT_ROOT/slurm/dgx/foundation_summary.slurm"
fi
summary_job="$(
    sbatch --parsable \
        --dependency="afterok:$dependency" \
        --export="ALL,TIME_LAUNCH_ID=$launch_id" \
        "$summary_front"
)"
summary_job="${summary_job%%;*}"

echo "foundation summary submitted job_id=$summary_job dependency=afterok:$dependency"
echo "status: bash scripts/foundation_model_status.sh $cluster $launch_id"
