#!/bin/bash
set -euo pipefail

cd /home/user

ARTIFACT_DIR="/home/user/checkpoints/facet_planar"

echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting split evaluation"
echo "[run] artifact dir: ${ARTIFACT_DIR}"
find "${ARTIFACT_DIR}" -maxdepth 1 -type f -name '*.json' -printf '[run] artifact %f %s bytes\n' | sort

timeout 300s bash scripts/evaluate_train.sh "${ARTIFACT_DIR}"
echo "[run] train scores written"

timeout 300s bash scripts/evaluate_test.sh "${ARTIFACT_DIR}"
echo "[run] test scores written"
echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) finished split evaluation"
