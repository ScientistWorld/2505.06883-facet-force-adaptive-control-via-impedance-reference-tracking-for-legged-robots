#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u redteam/audit.py prepare null_rollouts_shortcut
set +e
bash scripts/evaluate_test.sh > redteam/null_rollouts_shortcut/test.stdout.log 2>&1
status=$?
set -e
python redteam/audit.py collect null_rollouts_shortcut "$status"
exit "$status"
