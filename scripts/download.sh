#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This reproduction uses procedural data, so this script only ensures the
# downloads directory exists for the standard workspace layout.

set -e

cd /home/user

mkdir -p data/downloads
echo "FACET planar benchmark uses procedural data only; no external downloads are required."
