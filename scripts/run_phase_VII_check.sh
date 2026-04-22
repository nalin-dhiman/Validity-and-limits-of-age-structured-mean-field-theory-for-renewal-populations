#!/bin/bash
set -e

echo "--- Phase VII: Regime & Scaling Check ---"

# 1. F-I Curve (Continuous)
echo "Generating F-I Curve..."
python src/simulate_single.py --config configs/base.yaml --out_dir results/single

# 2. Scaling Analysis
echo "Running Scaling Analysis (N=500..10000)..."
python src/run_scaling.py --config configs/base.yaml --out_dir results/scaling

echo "Phase VII Checks Complete."
