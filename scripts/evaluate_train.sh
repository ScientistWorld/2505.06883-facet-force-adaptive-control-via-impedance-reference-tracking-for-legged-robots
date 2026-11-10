#!/bin/bash
# Score every policy on disk on the visible train scenarios.
#
# Same protocol as the held-out test (eval/train mirrors eval/test with visible
# seeds). Writes scoring/scores_train.json.

set -euo pipefail

cd /home/user

CHECKPOINT_ROOT="${1:-/home/user/checkpoints}"
echo "[evaluate_train] root=${CHECKPOINT_ROOT}"
python -u eval/train/evaluate.py "${CHECKPOINT_ROOT}" --output scoring/scores_train.json
