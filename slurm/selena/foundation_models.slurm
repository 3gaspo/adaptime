#!/bin/bash
#SBATCH --job-name=stime_foundation
#SBATCH --array=0-15%4
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/%u/codes/improved/logs_selena/foundation_%A_%a.out
#SBATCH --error=/scratch/users/%u/codes/improved/logs_selena/foundation_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --partition=an
#SBATCH --qos=an_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --exclusive
#SBATCH --wckey=P12CU:DATASCIENCE

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
export TIME_LAUNCH_ID="${TIME_LAUNCH_ID:-selena_${SLURM_ARRAY_JOB_ID}}"
source "$PROJECT_ROOT/src/slurm/run_foundation_model.sh"
