#!/bin/bash

# One authoritative mapping shared by direct and Slurm foundation-model runs.
FOUNDATION_MODELS=(
    chronos_bolt
    chronos2
    tirex
    ts_icl
    seasonal_naive
)

FOUNDATION_RUNNERS=(
    run_chronos_bolt.sh
    run_chronos2.sh
    run_tirex.sh
    run_tsicl.sh
    run_seasonal_naive.sh
)

FOUNDATION_MODEL_COUNT="${#FOUNDATION_MODELS[@]}"
