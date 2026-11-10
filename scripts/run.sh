#!/bin/bash
set -euo pipefail

cd /home/user

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u redteam/audit.py batch
