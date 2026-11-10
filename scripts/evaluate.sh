#!/bin/bash
# Standard evaluation entry point (visible train scenarios).
#
# Uses only code in eval/: every TorchScript policy at
# <root>/<method>/policy.pt is rolled out by the evaluator in its own simulator
# copy, together with the evaluator-owned velocity baselines, and the results
# are written to scoring/scores.json (same metrics as scoring/reference.json).
#
# Usage: bash scripts/evaluate.sh [checkpoint_root]

set -euo pipefail

cd /home/user

mkdir -p scoring
CHECKPOINT_ROOT="${1:-/home/user/checkpoints}"
python -u eval/train/evaluate.py "${CHECKPOINT_ROOT}" --output scoring/scores.json
