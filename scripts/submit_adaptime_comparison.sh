#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
platform="${1:-}"
case "$platform" in
    dgx)
        front="$PROJECT_ROOT/slurm/dgx/adaptime_comparison.slurm"
        ;;
    selena)
        front="$PROJECT_ROOT/slurm/selena/adaptime_comparison_selena.slurm"
        ;;
    *)
        echo "Usage: $0 dgx|selena" >&2
        exit 2
        ;;
esac

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
sbatch --export=ALL "$front"
