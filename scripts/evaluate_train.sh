#!/bin/bash
# Score the visible train slice of saved rollout artifacts.

set -euo pipefail

cd /home/user

python eval/train/evaluate_rollouts.py "${1:-/home/user/checkpoints}" --output scoring/scores_train.json
