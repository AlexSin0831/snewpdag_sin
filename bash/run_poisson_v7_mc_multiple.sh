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

echo "Started batch at: $(date)"
echo "Job ID: ${JOB_ID}"
echo "Trials in this job: ${TRIALS_PER_JOB}"

for ((OFFSET = 0; OFFSET < TRIALS_PER_JOB; OFFSET++)); do
  TRIAL_ID=$((START_TRIAL + OFFSET))
  SEED=$((BASE_SEED + TRIAL_ID))
  TRIAL_START=$SECONDS

  echo "---------------------------------------"
  echo "Starting trial ID ${TRIAL_ID}, seed ${SEED} at $(date)"

  python3 snewpdag/data/Main_single_test_poisson_likelihood_v7.py \
    --seed "${SEED}" \
    "$@"

  echo "Finished trial ID ${TRIAL_ID} in $((SECONDS - TRIAL_START)) seconds"
done

echo "---------------------------------------"
echo "Finished batch at: $(date)"
echo "Total batch elapsed seconds: $((SECONDS - JOB_START))"
