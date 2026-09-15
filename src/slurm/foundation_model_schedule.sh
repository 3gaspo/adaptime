#!/bin/bash

# Adaptime uses these backbones and deliberately excludes TimesFM-3.
FOUNDATION_MODELS=(
    chronos_bolt
    chronos2
    ts_icl
    seasonal_naive
)

FOUNDATION_LEARNED_MODELS=(
    chronos_bolt
    chronos2
    ts_icl
)

FOUNDATION_RUNNERS=(
    run_chronos_bolt.sh
    run_chronos2.sh
    run_tsicl.sh
    run_seasonal_naive.sh
)
