#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
case "${1:-}" in
    dgx)
        front="$PROJECT_ROOT/slurm/dgx/tsrag_comparison.slurm"
        ;;
    selena)
        front="$PROJECT_ROOT/slurm/selena/tsrag_comparison_selena.slurm"
        ;;
    *)
        echo "Usage: $0 dgx|selena" >&2
        exit 2
        ;;
esac

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
sbatch --export=ALL "$front"
