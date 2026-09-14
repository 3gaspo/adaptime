#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
platform="${1:-}"
afterok_job="${2:-}"
case "$platform" in
    dgx)
        front="$PROJECT_ROOT/slurm/dgx/adaptime_comparison.slurm"
        ;;
    selena)
        front="$PROJECT_ROOT/slurm/selena/adaptime_comparison_selena.slurm"
        ;;
    *)
        echo "Usage: $0 dgx|selena [AFTEROK_JOB_ID]" >&2
        exit 2
        ;;
esac
if [ "$#" -gt 2 ] || { [ -n "$afterok_job" ] && ! [[ "$afterok_job" =~ ^[0-9]+$ ]]; }; then
    echo "Usage: $0 dgx|selena [AFTEROK_JOB_ID]" >&2
    exit 2
fi

dependency=()
[ -z "$afterok_job" ] || dependency=(--dependency="afterok:$afterok_job")

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
workflow_id="${TIME_LAUNCH_ID:-${platform}_rolling_ridge_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
prepare_job="$(sbatch --parsable --job-name=rolling_prepare "${dependency[@]}" "--export=ALL,TIME_LAUNCH_ID=${workflow_id}_prepare,ADAPTIME_METHOD=vanilla,ADAPTIME_STAGE=prepare" "$front")"
prepare_job="${prepare_job%%;*}"
vanilla_job="$(sbatch --parsable --job-name=rolling_vanilla --dependency="afterok:$prepare_job" "--export=ALL,TIME_LAUNCH_ID=${workflow_id}_vanilla,ADAPTIME_METHOD=vanilla,ADAPTIME_STAGE=vanilla" "$front")"
vanilla_job="${vanilla_job%%;*}"
rolling_job="$(sbatch --parsable --job-name=rolling_ridge --dependency="afterok:$vanilla_job" "--export=ALL,TIME_LAUNCH_ID=${workflow_id}_rolling,ADAPTIME_METHOD=rolling_y_ridge_horizon,ADAPTIME_STAGE=pipeline" "$front")"
rolling_job="${rolling_job%%;*}"
printf 'workflow=%s prepare=%s vanilla=%s rolling=%s\n' \
    "$workflow_id" "$prepare_job" "$vanilla_job" "$rolling_job"
