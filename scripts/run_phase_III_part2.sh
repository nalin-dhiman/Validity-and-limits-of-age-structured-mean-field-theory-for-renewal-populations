#!/bin/bash
set -e

# Helper to update yaml
update_yaml() {
    python -c "import yaml; d=yaml.safe_load(open('$1')); $2; yaml.dump(d, open('$3','w'))"
}

echo "--- Phase III Part 2: Scaling & Universality ---"

# 1. Finite-Size Scaling (Fig A8)
echo "[1/2] Running Finite-Size Scaling..."
python src/run_scaling.py --config configs/base.yaml --out_dir results/scaling

# Plot Scaling
# run_scaling.py saves scaling_results.csv and scaling_fit.txt.
# We need to plot it with the fit.
python -c "import pandas as pd; import matplotlib.pyplot as plt; import numpy as np; 
df=pd.read_csv('results/scaling/scaling_results.csv'); 
fit_lines = open('results/scaling/scaling_fit.txt').readlines();
alpha = float(fit_lines[0].split(': ')[1]);
popt_str = fit_lines[1].split(': ')[1].strip('[]\n '); 
# Parse popt manualy or just use alpha
# popt = [A, alpha]
# Let's re-fit for plot or just plot line slope
plt.figure(); plt.loglog(df['N'], df['rmse'], 'o', label='Data'); 
# Reference line
fit_val = df['rmse'].iloc[0] * (df['N']/df['N'].iloc[0])**(-alpha)
plt.loglog(df['N'], fit_val, '--', label=f'Fit $\\alpha={alpha:.2f}$');
plt.loglog(df['N'], df['rmse'].iloc[0] * (df['N']/df['N'].iloc[0])**(-0.5), ':', label='Ref $N^{-1/2}$');
plt.xlabel('N'); plt.ylabel('RMSE'); plt.title('Finite-Size Scaling'); plt.legend();
plt.grid(True, which='both', ls='-');
plt.savefig('results/figures/Fig_A8_Scaling.png')"

# 2. Universality (Softplus)
echo "[2/2] Running Universality Test (Softplus)..."
# Create config
update_yaml "configs/base.yaml" "d['spike_gen']['type']='softplus'" "configs/softplus.yaml"

# Run Failure Map (Phase Diagram) with Softplus
# Output dir separate to avoid overwrite? Or same?
# Prompt asks for "Fig_A10_Universality_*" overlays.
# If we generate Fig_A7 (failure map) for softplus, we can rename it.
python src/run_failure_map.py --config configs/softplus.yaml --out_dir results/universality

# Copy Result
cp results/universality/Fig_A7_FailureMap.png results/figures/Fig_A10_Universality_Softplus.png

echo "Phase III Part 2 Complete."
