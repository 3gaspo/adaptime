#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
platform="${1:-}"
case "$platform" in
    dgx)
        ridge_front="$PROJECT_ROOT/slurm/dgx/adaptime_comparison.slurm"
        tsrag_front="$PROJECT_ROOT/slurm/dgx/tsrag_comparison.slurm"
        ;;
    selena)
        ridge_front="$PROJECT_ROOT/slurm/selena/adaptime_comparison_selena.slurm"
        tsrag_front="$PROJECT_ROOT/slurm/selena/tsrag_comparison_selena.slurm"
        ;;
    *)
        echo "Usage: $0 dgx|selena" >&2
        exit 2
        ;;
esac

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
prepare_job="$(sbatch --parsable --export=ALL,ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=prepare "$ridge_front")"
prepare_job="${prepare_job%%;*}"
tsrag_job="$(sbatch --parsable --dependency="afterok:$prepare_job" --export=ALL,ADAPTIME_METHOD=tsrag,ADAPTIME_STAGE=pipeline "$tsrag_front")"
tsrag_job="${tsrag_job%%;*}"
ridge_job="$(sbatch --parsable --dependency="afterok:$prepare_job" --export=ALL,ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=pipeline "$ridge_front")"
ridge_job="${ridge_job%%;*}"
report_dependency="afterok:$ridge_job:$tsrag_job"
report_job="$(sbatch --parsable --dependency="$report_dependency" --export=ALL,ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=report "$ridge_front")"
report_job="${report_job%%;*}"
printf 'prepare=%s ridge=%s tsrag=%s report=%s\n' "$prepare_job" "$ridge_job" "$tsrag_job" "$report_job"
