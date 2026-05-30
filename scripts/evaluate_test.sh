#!/bin/bash
# Score the held-out test slice of saved rollout artifacts.

set -euo pipefail

cd /home/user

python eval/test/evaluate_rollouts.py "${1:-/home/user/checkpoints}" --output scoring/scores_test.json
