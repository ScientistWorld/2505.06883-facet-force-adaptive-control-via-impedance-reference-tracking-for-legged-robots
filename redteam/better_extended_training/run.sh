#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u redteam/audit.py prepare better_extended_training
set +e
bash scripts/evaluate_test.sh > redteam/better_extended_training/test.stdout.log 2>&1
status=$?
set -e
python redteam/audit.py collect better_extended_training "$status"
exit "$status"
