#!/bin/bash
# Score the visible train slice of saved rollout artifacts.

set -euo pipefail

cd /home/user

INPUT_DIR="${1:-/home/user/checkpoints/facet_planar}"
echo "[evaluate_train] input=${INPUT_DIR}"
python -u eval/train/evaluate_rollouts.py "${INPUT_DIR}" --output scoring/scores_train.json
