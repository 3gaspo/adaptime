#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash sync_code_to_selena.sh [--dry-run]" >&2
}

RSYNC_OPTIONS=()
DRY_RUN=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) RSYNC_OPTIONS+=(--dry-run); DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
NNI_FILE="${TIME_NNI_FILE:-$HOME/codes/.secrets/nni}"
if [ ! -f "$NNI_FILE" ]; then
    echo "ERROR: missing $NNI_FILE" >&2
    exit 1
fi

NNI="$(sed -n '1p' "$NNI_FILE" | tr -d '[:space:]')"
nni="${NNI,,}"
if [[ ! "$nni" =~ ^[a-z][a-z0-9_-]*$ ]]; then
    echo "ERROR: $NNI_FILE must contain one valid NNI" >&2
    exit 1
fi

SELENA_HOST="${TIME_SELENA_HOST:-$nni@selena.hpc.edf.fr}"
DESTINATION="${TIME_SELENA_CODE_ROOT:-$SELENA_HOST:~/codes/$PROJECT_NAME/}"
SCRATCH_PROJECT_ROOT="${TIME_SELENA_SCRATCH_ROOT:-/scratch/users/$nni/codes/$PROJECT_NAME}"

echo "Synchronizing $PROJECT_NAME code from DGX to Selena..."
rsync -rlptz --delete-delay --itemize-changes --partial --info=progress2 \
    "${RSYNC_OPTIONS[@]}" \
    --exclude='.git/' \
    --exclude='.env' \
    --exclude='.venv/' \
    --exclude='.secrets/' \
    --exclude='AGENTS.md' \
    --exclude='FUTURE_WORK.md' \
    --exclude='PENDING_UPDATES.md' \
    --exclude='CLUSTER_STATUS.txt' \
    --exclude='docs/INTERNAL_WORKFLOW.md' \
    --exclude='datasets/' \
    --exclude='weights/' \
    --exclude='outputs/' \
    --exclude='logs/' \
    --include='outputs_selena/' \
    --include='outputs_selena/.gitkeep' \
    --exclude='outputs_selena/***' \
    --include='logs_selena/' \
    --include='logs_selena/.gitkeep' \
    --exclude='logs_selena/***' \
    --exclude='experiments/Kairos/' \
    --exclude='experiments/granite-tsfm/' \
    --exclude='experiments/timesfm_*/' \
    "$PROJECT_ROOT/" \
    "$DESTINATION"

if [ "$DRY_RUN" = true ]; then
    echo "PREVIEW: no files were transferred or deleted."
    exit 0
fi

ssh "$SELENA_HOST" \
    "mkdir -p '$SCRATCH_PROJECT_ROOT/outputs_selena' '$SCRATCH_PROJECT_ROOT/logs_selena'"

echo "SUCCESS: Selena's $PROJECT_NAME code matches DGX."
echo "Selena results: $SCRATCH_PROJECT_ROOT/outputs_selena/results"
echo "Selena logs: $SCRATCH_PROJECT_ROOT/logs_selena"
