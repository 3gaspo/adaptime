#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
case "${1:-}" in
    dgx)
        front="$PROJECT_ROOT/slurm/dgx/tsrag_comparison.slurm"
        ;;
    selena)
        front="$PROJECT_ROOT/slurm/selena/tsrag_comparison_selena.slurm"
        ;;
    *)
        echo "Usage: $0 dgx|selena" >&2
        exit 2
        ;;
esac

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
prepare_job="$(sbatch --parsable --export=ALL,ADAPTIME_STAGE=prepare "$front")"
prepare_job="${prepare_job%%;*}"
tsrag_job="$(sbatch --parsable --dependency="afterok:$prepare_job" --export=ALL,ADAPTIME_STAGE=pipeline "$front")"
tsrag_job="${tsrag_job%%;*}"
if [ -n "${ADAPTIME_RIDGE_RESULTS_PATH:-}" ]; then
    report_job="$(sbatch --parsable --dependency="afterok:$tsrag_job" --export=ALL,ADAPTIME_STAGE=report "$front")"
    report_job="${report_job%%;*}"
else
    report_job=not_requested
fi
printf 'prepare=%s tsrag=%s report=%s\n' "$prepare_job" "$tsrag_job" "$report_job"
