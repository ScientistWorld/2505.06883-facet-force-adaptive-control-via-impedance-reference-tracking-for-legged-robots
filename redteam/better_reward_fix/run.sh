#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u redteam/audit.py prepare better_reward_fix
set +e
bash scripts/evaluate_test.sh > redteam/better_reward_fix/test.stdout.log 2>&1
status=$?
set -e
python redteam/audit.py collect better_reward_fix "$status"
exit "$status"
