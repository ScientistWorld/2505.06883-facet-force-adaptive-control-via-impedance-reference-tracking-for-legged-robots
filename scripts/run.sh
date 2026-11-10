#!/bin/bash
set -euo pipefail

cd /home/user

echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting core reproduction"
bash scripts/reproduce.sh
echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) finished core reproduction"
