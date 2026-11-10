#!/bin/bash
# Velocity-tracking baselines.
#
# The evaluators run their own copies of both baselines on every call (they are
# the fixed comparison side), so the official baseline numbers come from
# scripts/evaluate*.sh. This script additionally exports the same controllers
# as TorchScript policies (an example of the policy interface) and scores them
# on the visible train scenarios, which must reproduce the evaluator's own
# baseline rows.

set -euo pipefail

cd /home/user

python -m baseline.velocity_controllers --output-root outputs/baselines
python -u eval/train/evaluate.py outputs/baselines --output outputs/baselines/scores_train.json
