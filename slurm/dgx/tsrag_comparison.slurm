#!/bin/bash
#SBATCH --job-name=adaptime_tsrag
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-dgx_tsrag_${SLURM_JOB_ID}}"
export TSRAG_RIDGE_OUTPUT_ROOT="${TSRAG_RIDGE_OUTPUT_ROOT:-$TIME_OUTPUTS/adaptime}"
source "$PROJECT_ROOT/src/slurm/run_tsrag_comparison.sh"
