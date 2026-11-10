#!/bin/bash
# Final hardening job (resumable after preemption: trained policies and finished
# scorings on disk are reused).
#   1. consistency checks between data/, method/, eval/train and eval/test
#   2. rescore every policy trained in the previous round, the controller and
#      shortcut probes and the brute-force grid under the fixed evaluators
#   3. train the paper-size-network configuration (5 seeds) and its 2x-budget,
#      open-loop and untrained variants
#   4. choose the reference configuration (pre-registered rule, train slice),
#      ship its median seed as checkpoints/facet, score the probes wrapping it
#   5. analysis (ranks, seed noise, paired bootstrap comparisons)
#   6. official scores from the shipped reference and the evaluator baselines
set -uo pipefail

cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${HARDEN_MODE:-full}"
echo "[run] $(date -u +%FT%TZ) mode=${MODE}"
python -c "import os, torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'affinity', len(os.sched_getaffinity(0)))"

python -u redteam/hardened/harness.py checks || echo "[run] checks failed"
python -u redteam/hardened/harness.py score-existing --mode "${MODE}" || echo "[run] score-existing failed"
python -u redteam/hardened/harness.py analyze --mode "${MODE}" || echo "[run] interim analyze failed"
echo "[run] $(date -u +%FT%TZ) rescoring done"
python -u redteam/hardened/harness.py train-paper --mode "${MODE}" || echo "[run] train-paper failed"
echo "[run] $(date -u +%FT%TZ) training done"
python -u redteam/hardened/harness.py finalize --mode "${MODE}" || echo "[run] finalize failed"
python -u redteam/hardened/harness.py analyze --mode "${MODE}" || echo "[run] analyze failed"

set -e
bash scripts/baseline.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare --post-design || true
echo "[run] $(date -u +%FT%TZ) done"
