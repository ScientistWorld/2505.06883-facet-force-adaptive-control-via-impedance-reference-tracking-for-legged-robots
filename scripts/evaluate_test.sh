#!/bin/bash
# Score every policy on disk on the held-out test scenarios.
#
# Each method is a TorchScript policy at <root>/<method>/policy.pt (default root
# /home/user/checkpoints). The evaluator rolls every policy out in its own
# simulator copy against its own baselines and writes scoring/scores_test.json.
# Run with no arguments.

set -euo pipefail

cd /home/user

CHECKPOINT_ROOT="${1:-/home/user/checkpoints}"
echo "[evaluate_test] root=${CHECKPOINT_ROOT}"
python -u eval/test/evaluate.py "${CHECKPOINT_ROOT}" --output scoring/scores_test.json
