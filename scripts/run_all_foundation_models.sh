#!/bin/bash
# Run every foundation-model reproduction runner, then record one summary table.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/src/slurm/runtime_paths.sh"
source "$ROOT_DIR/src/slurm/foundation_model_runners.sh"

cd "$ROOT_DIR"
for runner in "${FOUNDATION_RUNNERS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $runner"
    bash "$SCRIPT_DIR/$runner"
done

"${SUMMARY_PYTHON:-python}" "$SCRIPT_DIR/compute_foundation_summary.py"
