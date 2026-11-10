#!/bin/bash
# Final hardening job (resumable after preemption):
#   1. consistency checks between data/, method/, eval/train and eval/test
#   2. train 5 reference seeds + quality variants (single-threaded, parallel),
#      score them, ship the median reference seed as checkpoints/facet, build
#      and score the shortcut probes through the unmodified hidden test
#   3. analysis (ranks, seed noise, paired bootstrap comparisons)
#   4. official scores from the shipped reference and the evaluator baselines
set -uo pipefail

cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${HARDEN_MODE:-full}"
echo "[run] $(date -u +%FT%TZ) mode=${MODE}"
python -c "import os, torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'affinity', len(os.sched_getaffinity(0)))"

python -u redteam/hardened/harness.py checks || echo "[run] checks failed"
python -u redteam/hardened/harness.py run --mode "${MODE}" || echo "[run] harness run failed"
python -u redteam/hardened/harness.py analyze --mode "${MODE}" || echo "[run] analyze failed"

set -e
bash scripts/baseline.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare --post-design || true
echo "[run] $(date -u +%FT%TZ) done"
