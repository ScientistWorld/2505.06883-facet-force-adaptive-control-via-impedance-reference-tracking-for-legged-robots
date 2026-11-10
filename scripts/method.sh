#!/bin/bash
# Train the paper's method (reduced-scale FACET teacher/student PPO) and export
# the deployable student to checkpoints/facet/policy.pt.
#
# Usage: bash scripts/method.sh   (FORCE=1 retrains even if a policy exists)

set -euo pipefail

cd /home/user

EXTRA=()
if [ "${FORCE:-0}" = "1" ]; then EXTRA+=(--force); fi
python -u -m method.run_facet \
  --output-dir checkpoints/facet \
  --seed "${SEED:-7}" \
  --teacher-updates "${TEACHER_UPDATES:-160}" \
  --student-updates "${STUDENT_UPDATES:-40}" \
  --n-envs "${N_ENVS:-256}" \
  --rollout-steps "${ROLLOUT_STEPS:-96}" \
  "${EXTRA[@]}"
