#!/bin/bash

# One capability mapping shared by direct and Slurm foundation-model runs.
FOUNDATION_SUPPORTED_MODELS=(
    chronos_bolt
    chronos2
    timesfm3
    ts_icl
    seasonal_naive
)

FOUNDATION_SUPPORTED_RUNNERS=(
    run_chronos_bolt.sh
    run_chronos2.sh
    run_timesfm3.sh
    run_tsicl.sh
    run_seasonal_naive.sh
)

FOUNDATION_MODELS=()
FOUNDATION_LEARNED_MODELS=()
FOUNDATION_RUNNERS=()

foundation_project_root="${PROJECT_ROOT:-${ROOT_DIR:-$(pwd)}}"
schedule_file="${TIME_FOUNDATION_SCHEDULE:-$foundation_project_root/src/slurm/foundation_model_schedule.sh}"
if [ -f "$schedule_file" ]; then
    source "$schedule_file"
fi
FOUNDATION_MODEL_COUNT="${#FOUNDATION_MODELS[@]}"

require_foundation_schedule() {
    if [ "$FOUNDATION_MODEL_COUNT" -eq 0 ]; then
        echo "No active foundation schedule; create src/slurm/foundation_model_schedule.sh" >&2
        return 2
    fi
    if [ "${#FOUNDATION_RUNNERS[@]}" -ne "$FOUNDATION_MODEL_COUNT" ]; then
        echo "Foundation schedule model and runner counts differ" >&2
        return 2
    fi
}
