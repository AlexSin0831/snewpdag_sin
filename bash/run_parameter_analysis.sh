#!/bin/bash
set -euo pipefail

JOB_ID=$1
TRIALS_PER_JOB=$2
shift 2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="${PROJECT_ROOT}/venv/bin/activate"
BASE_SEED=1000
START_TRIAL=$((JOB_ID * TRIALS_PER_JOB))
JOB_START=$SECONDS

cd "${PROJECT_ROOT}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtual environment not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

mkdir -p output logs .mplconfig
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplconfig"

for ((OFFSET = 0; OFFSET < TRIALS_PER_JOB; OFFSET++)); do
  TRIAL_ID=$((START_TRIAL + OFFSET))
  SEED=$((BASE_SEED + TRIAL_ID))

  python3 "${PROJECT_ROOT}/snewpdag/data/Parameter_analysis.py" \
    --seed "${SEED}" \
    "$@"
done
