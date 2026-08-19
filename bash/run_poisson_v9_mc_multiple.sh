#!/bin/bash
set -euo pipefail

JOB_ID=$1
TRIALS_PER_JOB=$2
shift 2

BASE_SEED=1000
START_TRIAL=$((JOB_ID * TRIALS_PER_JOB))
JOB_START=$SECONDS

cd /data/snoplus3/sin2/snewpdag

source /data/snoplus3/sin2/snewpdag/venv/bin/activate

mkdir -p output logs .mplconfig
export MPLCONFIGDIR=/data/snoplus3/sin2/snewpdag/.mplconfig

for ((OFFSET = 0; OFFSET < TRIALS_PER_JOB; OFFSET++)); do
  TRIAL_ID=$((START_TRIAL + OFFSET))
  SEED=$((BASE_SEED + TRIAL_ID))

  python3 snewpdag/data/Main_single_test_poisson_likelihood_v9.py \
    --seed "${SEED}" \
    "$@"
done


