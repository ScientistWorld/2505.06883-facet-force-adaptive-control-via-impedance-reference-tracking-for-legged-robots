#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u redteam/audit.py prepare reference_seed7
set +e
bash scripts/evaluate_test.sh > redteam/reference_seed7/test.stdout.log 2>&1
status=$?
set -e
python redteam/audit.py collect reference_seed7 "$status"
exit "$status"
