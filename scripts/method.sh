#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.run_facet \
  --episodes "${EPISODES:-96}" \
  --output-dir checkpoints/facet_planar \
  --teacher-updates "${TEACHER_UPDATES:-60}" \
  --student-updates "${STUDENT_UPDATES:-30}" \
  --n-envs "${N_ENVS:-128}" \
  --rollout-steps "${ROLLOUT_STEPS:-96}"
