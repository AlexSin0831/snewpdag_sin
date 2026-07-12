#!/bin/bash
set -e

cd /data/snoplus3/sin/snewpdag

source /data/snoplus3/sin/snewpdag/venv/bin/activate

mkdir -p output logs .mplconfig
export MPLCONFIGDIR=/data/snoplus3/sin/snewpdag/.mplconfig

echo "Started at: $(date)"
SECONDS=0

python snewpdag/data/Main_single_test_poisson_likelihood_v3.py "$@"

echo "Finished at: $(date)"
echo "Elapsed seconds: $SECONDS"
