#!/bin/bash

set -euo pipefail

usage() {
    echo "usage: bash clear_selena_artifacts.sh dgx|selena" >&2
}

platform="${1:-}"
case "$platform" in
    dgx|selena) ;;
    *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$SCRIPT_DIR"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"

if [ "$platform" = dgx ]; then
    artifact_roots=(
        "$PROJECT_ROOT/logs/selena"
        "$PROJECT_ROOT/outputs/selena"
    )
else
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
    scratch_project_root="/scratch/users/$nni/codes/$PROJECT_NAME"
    artifact_roots=(
        "$scratch_project_root/logs"
        "$scratch_project_root/outputs"
    )
fi

for artifact_root in "${artifact_roots[@]}"; do
    if [ ! -d "$artifact_root" ] || [ -L "$artifact_root" ]; then
        echo "ERROR: expected a real artifact directory: $artifact_root" >&2
        exit 1
    fi
done

for artifact_root in "${artifact_roots[@]}"; do
    (
        cd "$artifact_root"
        shopt -s nullglob
        entries=(*)
        if [ "${#entries[@]}" -gt 0 ]; then
            rm -r -- "${entries[@]}"
        fi
    )
    echo "cleared $artifact_root"
done
