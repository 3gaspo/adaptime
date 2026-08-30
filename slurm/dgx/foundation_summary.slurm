#!/bin/bash
#SBATCH --job-name=time_summary
#SBATCH --gres=gpu:1
#SBATCH --output=logs/foundation_summary_%j.out
#SBATCH --error=logs/foundation_summary_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/summarize_foundation_models.sh"
