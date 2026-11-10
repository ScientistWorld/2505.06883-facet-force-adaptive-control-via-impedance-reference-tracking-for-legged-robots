#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u redteam/audit.py prepare impossible_flags_shortcut
set +e
bash scripts/evaluate_test.sh > redteam/impossible_flags_shortcut/test.stdout.log 2>&1
status=$?
set -e
python redteam/audit.py collect impossible_flags_shortcut "$status"
exit "$status"
