#!/bin/bash
# Train the paper's method (reduced-scale FACET teacher/student PPO) and export
# the deployable student to checkpoints/facet/policy.pt.
#
# Budget: 600 teacher + 150 student PPO updates of 256 envs x 96 steps (14.7M +
# 3.7M env steps, the paper's 4:1 split at reduced scale). The shipped reference
# is training seed 27, the median of five seeds (7/17/27/37/47); training is
# single-threaded and deterministic for a given seed on a given CPU type.
#
# Usage: bash scripts/method.sh   (FORCE=1 retrains even if a policy exists)

set -euo pipefail

cd /home/user

EXTRA=()
if [ "${FORCE:-0}" = "1" ]; then EXTRA+=(--force); fi
python -u -m method.run_facet \
  --output-dir checkpoints/facet \
  --seed "${SEED:-27}" \
  --teacher-updates "${TEACHER_UPDATES:-600}" \
  --student-updates "${STUDENT_UPDATES:-150}" \
  --n-envs "${N_ENVS:-256}" \
  --rollout-steps "${ROLLOUT_STEPS:-96}" \
  "${EXTRA[@]}"
