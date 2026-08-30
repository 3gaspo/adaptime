#!/bin/bash

# One authoritative mapping shared by direct and Slurm foundation-model runs.
FOUNDATION_MODELS=(
    chronos_bolt
    chronos2
    kairos
    litespecformer
    moirai
    moirai2
    patchtst_fm
    sundial
    timesfm1
    timesfm2
    timesfm2p5
    timesfm3
    tirex
    toto
    tsicl
    visiontspp
)

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

FOUNDATION_MODEL_COUNT="${#FOUNDATION_MODELS[@]}"
