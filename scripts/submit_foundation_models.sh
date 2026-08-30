#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash scripts/submit_foundation_models.sh dgx" >&2
    echo "Selena: submit slurm/selena/foundation_models.slurm directly with sbatch." >&2
}

cluster="${1:-}"
case "$cluster" in
    dgx) ;;
    *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
mkdir -p "$TIME_LOGS"

evaluation_job="$(sbatch --parsable "$PROJECT_ROOT/slurm/$cluster/foundation_models.slurm")"
evaluation_job="${evaluation_job%%;*}"
launch_id="${cluster}_${evaluation_job}"
summary_job="$(
    sbatch --parsable \
        --dependency="afterok:$evaluation_job" \
        --export="ALL,TIME_LAUNCH_ID=$launch_id" \
        "$PROJECT_ROOT/slurm/$cluster/foundation_summary.slurm"
)"
summary_job="${summary_job%%;*}"

echo "foundation evaluation array submitted job_id=$evaluation_job launch_id=$launch_id"
echo "foundation summary submitted job_id=$summary_job dependency=afterok:$evaluation_job"
echo "status: bash scripts/foundation_model_status.sh dgx $launch_id"
