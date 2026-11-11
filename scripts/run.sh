#!/bin/bash
# Reproduce the reference end to end (offline, in the container from environment/):
# train the FACET reference (seed 17; skipped if checkpoints/facet/policy.pt
# exists, FORCE=1 retrains), export and score the baselines, score every policy
# on disk on the visible train and held-out test scenarios, and validate.
#
# The hardening audit (all variants, probes and analyses in HARDENING.md) is
# rerun with redteam/hardened/harness.py:
#   checks | score-existing | train-paper | finalize | analyze
set -euo pipefail

cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[run] $(date -u +%FT%TZ) start"
bash scripts/reproduce.sh
echo "[run] $(date -u +%FT%TZ) done"
