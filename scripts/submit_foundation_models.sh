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

if [ "$cluster" = selena ]; then
    seasonal_front="$PROJECT_ROOT/slurm/selena/foundation_models/seasonal_naive_selena.slurm"
else
    seasonal_front="$PROJECT_ROOT/slurm/dgx/foundation_models/seasonal_naive.slurm"
fi
seasonal_job="$(
    sbatch --parsable \
        --export="ALL,TIME_LAUNCH_ID=$launch_id" \
        "$seasonal_front"
)"
seasonal_job="${seasonal_job%%;*}"
model_jobs=("$seasonal_job")
echo "foundation model submitted model=seasonal_naive job_id=$seasonal_job launch_id=$launch_id"

for model in "${FOUNDATION_MODELS[@]}"; do
    [ "$model" != seasonal_naive ] || continue
    if [ "$cluster" = selena ]; then
        front="$PROJECT_ROOT/slurm/selena/foundation_models/${model}_selena.slurm"
    else
        front="$PROJECT_ROOT/slurm/dgx/foundation_models/${model}.slurm"
    fi
    job_id="$(
        sbatch --parsable \
            --dependency="afterok:$seasonal_job" \
            --export="ALL,TIME_LAUNCH_ID=$launch_id" \
            "$front"
    )"
    job_id="${job_id%%;*}"
    model_jobs+=("$job_id")
    echo "foundation model submitted model=$model job_id=$job_id launch_id=$launch_id dependency=afterok:$seasonal_job"
done

dependency="$(IFS=:; echo "${model_jobs[*]}")"
if [ "$cluster" = selena ]; then
    summary_front="$PROJECT_ROOT/slurm/selena/foundation_summary_selena.slurm"
else
    summary_front="$PROJECT_ROOT/slurm/dgx/foundation_summary.slurm"
fi
summary_job="$(
    sbatch --parsable \
        --dependency="afterany:$dependency" \
        --export="ALL,TIME_LAUNCH_ID=$launch_id" \
        "$summary_front"
)"
summary_job="${summary_job%%;*}"

echo "foundation summary and feature plot submitted job_id=$summary_job dependency=afterany:$dependency"
echo "status: bash scripts/foundation_model_status.sh $cluster $launch_id"
