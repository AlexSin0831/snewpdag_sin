#!/bin/bash
set -euo pipefail

cd /Users/alexsin/Desktop/SURE/snewpdag
mkdir -p output 

python -m snewpdag.trials.Simple Control -n 2 | \
env MPLCONFIGDIR=/private/tmp/snewpdag-mpl \
    TAG=poisson_skymap_single \
    MODEL=s27 \
    SPECIES=ibd \
    CHANNEL=ibd \
    RA=60.0 \
    DEC=30.0 \
    PIXEL8=18 \
    PIXEL32=297 \
    EPOCH_BASE=1635744156.328 \
    YIELD_IC=660000 \
    YIELD_SK=7800 \
    YIELD_SNOPLUS=280 \
    YIELD_JUNO=7200 \
    BG_IC=1476000 \
    BG_SK=0.1 \
    BG_SNOPLUS=0.001 \
    BG_JUNO=0.0015 \
    SEN_IC=94285 \
    SEN_SK=1114 \
    SEN_SNOPLUS=40 \
    SEN_JUNO=1029 \
    DISTANCE=10 \
    BIN_WIDTH=0.002 \
    WINDOW_START=-0.5 \
    WINDOW_SIZE=2.0 \
    SIGMA_GND=0.005 \
    IMPACT_RANGE=2.0 \
    CACHE_HIST1=True \
python -m snewpdag --jsonlines --log INFO \
    snewpdag/data/fe7.csv
