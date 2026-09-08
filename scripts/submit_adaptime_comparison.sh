#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
platform="${1:-}"
case "$platform" in
    dgx)
        family_front="$PROJECT_ROOT/slurm/dgx/adaptime_comparison.slurm"
        ;;
    selena)
        family_front="$PROJECT_ROOT/slurm/selena/adaptime_comparison_selena.slurm"
        ;;
    *)
        echo "Usage: $0 dgx|selena" >&2
        exit 2
        ;;
esac

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
prepare_job="$(sbatch --parsable --export=ALL,ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=prepare "$family_front")"
prepare_job="${prepare_job%%;*}"
family_job="$(sbatch --parsable --dependency="afterok:$prepare_job" --export=ALL,ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=pipeline "$family_front")"
family_job="${family_job%%;*}"
report_job="$(sbatch --parsable --dependency="afterok:$family_job" --export=ALL,ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=report "$family_front")"
report_job="${report_job%%;*}"
printf 'prepare=%s family=%s report=%s\n' "$prepare_job" "$family_job" "$report_job"
