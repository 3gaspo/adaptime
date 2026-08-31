#!/bin/bash
#SBATCH --job-name=time_foundation
#SBATCH --array=0-4%4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/foundation_%A_%a.out
#SBATCH --error=logs/foundation_%A_%a.err
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
export TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-dgx_${SLURM_ARRAY_JOB_ID}}"
source "$PROJECT_ROOT/src/slurm/run_foundation_model.sh"
