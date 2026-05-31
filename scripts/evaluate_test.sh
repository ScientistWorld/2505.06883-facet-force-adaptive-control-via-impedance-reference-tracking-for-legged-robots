#!/bin/bash
# Score the held-out test slice of saved rollout artifacts.

set -euo pipefail

cd /home/user

INPUT_DIR="${1:-/home/user/checkpoints/facet_planar}"
echo "[evaluate_test] input=${INPUT_DIR}"
python -u eval/test/evaluate_rollouts.py "${INPUT_DIR}" --output scoring/scores_test.json
