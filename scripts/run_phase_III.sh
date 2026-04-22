#!/bin/bash
set -e
mkdir -p results/figures results/tables results/sweep

echo "--- Phase III: APS PRE Upgrades ---"

# 1. Update Config (Already done)

# 2. Hazard Field Visualization (Fig A12)
# echo "[1/4] Generating Hazard Field..."
# Clean run with online PDE and saving snapshot
# python src/simulate_population.py --config configs/base.yaml --outfile results/hazard_run.npz --online-pde

# Plot Hazard Snapshot (Already generated)
# python -c "import numpy as np; import matplotlib.pyplot as plt; 
# data = np.load('results/hazard_run.npz'); 
# rho = data['rho_snapshot']; r_grid = data['r_grid']; 
# plt.figure(); plt.plot(r_grid, rho); 
# plt.xlabel('Age (s)'); plt.ylabel('Hazard Rate (Hz)'); 
# plt.title('Empirical Hazard Closure (Snapshot)'); 
# plt.savefig('results/figures/Fig_A12_HazardSnapshot.png')"

# 3. Phase Diagram (Regime Boundary)
echo "[2/4] Running Phase Diagram (Data-Driven)..."
python src/run_failure_map.py --config configs/base.yaml --out_dir results/sweep
cp results/sweep/Fig_A7_FailureMap.png results/figures/Fig_A9_Phase_Boundary.png
cp results/tables/failure_map.csv results/tables/phase_boundary.csv

# 4. Scaling (Updated)
# echo "[3/4] Running Scaling..."
# (Skip for speed unless requested, or run background)

echo "Phase III Complete."
