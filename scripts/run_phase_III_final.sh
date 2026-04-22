#!/bin/bash
set -e

# Helper to update yaml
update_yaml() {
    python -c "import yaml; d=yaml.safe_load(open('$1')); $2; yaml.dump(d, open('$3','w'))"
}

echo "--- Phase III Final Run: Phase Diagrams ---"

# 1. Standard Phase Diagram (Exponential)
echo "[1/2] Running Standard Phase Diagram (Exponential)..."
update_yaml "configs/base.yaml" "d['spike_gen']['type']='exponential'" "configs/base.yaml"
# Ensure duration 20s
update_yaml "configs/base.yaml" "d['duration']=20.0" "configs/base.yaml"

python src/run_failure_map.py --config configs/base.yaml --out_dir results/sweep
cp results/sweep/Fig_A7_FailureMap.png results/figures/Fig_A9_Phase_Boundary.png
cp results/tables/failure_map.csv results/tables/phase_boundary.csv

# 2. Universality Phase Diagram (Softplus)
echo "[2/2] Running Universality Phase Diagram (Softplus)..."
update_yaml "configs/base.yaml" "d['spike_gen']['type']='softplus'" "configs/softplus.yaml"
# Ensure duration 20s
update_yaml "configs/softplus.yaml" "d['duration']=20.0" "configs/softplus.yaml"

python src/run_failure_map.py --config configs/softplus.yaml --out_dir results/universality
cp results/universality/Fig_A7_FailureMap.png results/figures/Fig_A10_Universality_Softplus.png

echo "Phase III Final Complete."
