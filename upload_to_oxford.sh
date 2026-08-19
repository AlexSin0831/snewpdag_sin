#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OXFORD_DIR="pp:/data/snoplus3/sin2/snewpdag/"

exec rsync -avhP \
  --timeout=300 \
  -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='output/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='snewpdag/plugins/build/' \
  --exclude='snewpdag/plugins/build-linux/' \
  --exclude='snewpdag/plugins/_recursion_integral*.so' \
  --exclude='models/node_modules/' \
  --exclude='~$*' \
  "${PROJECT_DIR}/" \
  "${OXFORD_DIR}"
