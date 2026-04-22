#!/bin/bash
set -euo pipefail

RERUN_LIST="results/phase_scan/rerun_list.csv"

if [ ! -f "$RERUN_LIST" ]; then
    echo "No rerun list found at $RERUN_LIST. Nothing to do."
    exit 0
fi

echo "Found failures. Rerunning points from $RERUN_LIST..."
# We do NOT use --clean here, obviously
python3 src/run_phase_scan.py --out_dir results/phase_scan --rerun-list "$RERUN_LIST"
