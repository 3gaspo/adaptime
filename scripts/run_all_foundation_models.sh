#!/bin/bash
# Run every foundation-model reproduction runner, then record one summary table.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/src/slurm/runtime_paths.sh"

FOUNDATION_RUNNERS=(
    run_chronos_bolt.sh
    run_chronos2.sh
    run_kairos.sh
    run_litespecformer.sh
    run_moirai.sh
    run_moirai2.sh
    run_patchtst_fm.sh
    run_sundial.sh
    run_timesfm1.sh
    run_timesfm2.sh
    run_timesfm2p5.sh
    run_timesfm3.sh
    run_tirex.sh
    run_toto.sh
    run_tsicl.sh
    run_visiontspp.sh
)

cd "$ROOT_DIR"
for runner in "${FOUNDATION_RUNNERS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $runner"
    bash "$SCRIPT_DIR/$runner"
done

"${SUMMARY_PYTHON:-python}" "$SCRIPT_DIR/compute_foundation_summary.py"
