#!/bin/bash
# Reproduce the reduced-scale FACET results end to end.
#
# Trains the reference policy (checkpoints/facet/policy.pt), exports and scores
# the baselines, and evaluates every policy on disk on the train and held-out
# test scenarios. Assumes the container from environment/ and runs offline.

set -euo pipefail

cd /home/user

bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare --post-design
