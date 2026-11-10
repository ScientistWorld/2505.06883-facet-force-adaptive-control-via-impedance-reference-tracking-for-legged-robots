#!/bin/bash
# Hardening job: reference + baselines + official scores, consistency checks,
# then quality variants and shortcut probes through the hidden-test evaluator.
# HARDEN_MODE=smoke runs every code path with tiny training budgets.
set -euo pipefail

cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${HARDEN_MODE:-full}"
echo "[run] $(date -u +%FT%TZ) mode=${MODE} nproc=$(nproc)"
python -c "import os, torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'affinity', len(os.sched_getaffinity(0)))"

if [ "${MODE}" = "smoke" ]; then
  export TEACHER_UPDATES=2 STUDENT_UPDATES=1 N_ENVS=32 ROLLOUT_STEPS=16
  # Full-scale timing probe (3 updates at 256 envs x 96 steps).
  python -u -m method.run_facet --output-dir outputs/timing_probe --teacher-updates 2 --student-updates 1 --force
  python -u -m method.run_facet --output-dir outputs/timing_probe_mlp --teacher-updates 2 --student-updates 1 --estimator mlp --force
fi

# Resumable: an existing checkpoints/facet/policy.pt is kept (training itself
# also checkpoints every 10 updates) and already-scored cases are skipped.
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python -u redteam/hardened/harness.py checks
python -u redteam/hardened/harness.py run --mode "${MODE}"
python -u redteam/hardened/harness.py analyze --mode "${MODE}"
python validate.py --compare --post-design || true
echo "[run] $(date -u +%FT%TZ) done"
