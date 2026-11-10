#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u redteam/audit.py prepare simple_baseline
set +e
bash scripts/evaluate_test.sh > redteam/simple_baseline/test.stdout.log 2>&1
status=$?
set -e
python redteam/audit.py collect simple_baseline "$status"
exit "$status"
