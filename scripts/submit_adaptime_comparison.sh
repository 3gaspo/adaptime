#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
platform="${1:-}"
case "$platform" in
    dgx)
        family_front="$PROJECT_ROOT/slurm/dgx/adaptime_comparison.slurm"
        tsrag_front="$PROJECT_ROOT/slurm/dgx/tsrag_comparison.slurm"
        ;;
    selena)
        family_front="$PROJECT_ROOT/slurm/selena/adaptime_comparison_selena.slurm"
        tsrag_front="$PROJECT_ROOT/slurm/selena/tsrag_comparison_selena.slurm"
        ;;
    *)
        echo "Usage: $0 dgx|selena" >&2
        exit 2
        ;;
esac

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
workflow_id="${TIME_LAUNCH_ID:-${platform}_adaptime_unified_$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
prepare_launch_id="${workflow_id}_prepare"
seasonal_launch_id="${workflow_id}_seasonal"
vanilla_launch_id="${workflow_id}_vanilla"
ridge_launch_id="${workflow_id}_ridge"
tsrag_launch_id="${workflow_id}_tsrag"
report_launch_id="${workflow_id}_report"
prepare_job="$(sbatch --parsable --job-name=adaptime_prepare "--export=ALL,TIME_LAUNCH_ID=$prepare_launch_id,ADAPTIME_METHOD=vanilla,ADAPTIME_STAGE=prepare" "$family_front")"
prepare_job="${prepare_job%%;*}"
seasonal_job="$(sbatch --parsable --job-name=adaptime_seasonal --dependency="afterok:$prepare_job" "--export=ALL,TIME_LAUNCH_ID=$seasonal_launch_id,ADAPTIME_METHOD=seasonal_naive,ADAPTIME_STAGE=pipeline" "$family_front")"
seasonal_job="${seasonal_job%%;*}"
vanilla_job="$(sbatch --parsable --job-name=adaptime_vanilla --dependency="afterok:$seasonal_job" "--export=ALL,TIME_LAUNCH_ID=$vanilla_launch_id,ADAPTIME_METHOD=vanilla,ADAPTIME_STAGE=vanilla" "$family_front")"
vanilla_job="${vanilla_job%%;*}"
ridge_job="$(sbatch --parsable --job-name=adaptime_ridge --dependency="afterok:$vanilla_job" "--export=ALL,TIME_LAUNCH_ID=$ridge_launch_id,ADAPTIME_METHOD=ridge,ADAPTIME_STAGE=pipeline" "$family_front")"
ridge_job="${ridge_job%%;*}"
tsrag_job="$(sbatch --parsable --dependency="afterok:$vanilla_job" "--export=ALL,TIME_LAUNCH_ID=$tsrag_launch_id,ADAPTIME_METHOD=tsrag,ADAPTIME_STAGE=pipeline" "$tsrag_front")"
tsrag_job="${tsrag_job%%;*}"
report_job="$(sbatch --parsable --job-name=adaptime_report --dependency="afterok:$ridge_job:$tsrag_job" "--export=ALL,TIME_LAUNCH_ID=$report_launch_id,ADAPTIME_METHOD=unified,ADAPTIME_STAGE=report" "$family_front")"
report_job="${report_job%%;*}"
printf 'workflow=%s prepare=%s seasonal=%s vanilla=%s ridge=%s tsrag=%s report=%s\n' \
    "$workflow_id" "$prepare_job" "$seasonal_job" "$vanilla_job" "$ridge_job" "$tsrag_job" "$report_job"
