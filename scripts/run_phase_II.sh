#!/bin/bash
set -e
mkdir -p results/figures results/tables results/scaling results/phase_diagram

echo "--- Phase II: APS PRE Analysis ---"

# 1. Finite-Size Scaling (Fig A8)
echo "[1/4] Running Finite-Size Scaling..."
python src/run_scaling.py --config configs/base.yaml --out_dir results/scaling
# Plotting handled inside script? No, let's create a plotter if needed or add plot logic to script.
# run_scaling.py saves csv. Need to plot logging.
# Let's simple-plot it here via python snippet
python -c "import pandas as pd; import matplotlib.pyplot as plt; import numpy as np; 
df=pd.read_csv('results/scaling/scaling_results.csv'); 
plt.figure(); plt.loglog(df['N'], df['rmse'], 'o-', label='RMSE'); 
plt.xlabel('N'); plt.ylabel('RMSE'); plt.title('Finite-Size Scaling'); 
plt.savefig('results/figures/Fig_A8_Scaling.png')"

# 2. Breakdown Phase Diagram (Fig A9) - Exponential
echo "[2/4] Running Phase Diagram (Exponential)..."
# This might takes time. Reduced grid in script?
python src/run_phase_diagram.py --config configs/base.yaml --out_dir results/phase_diagram --hazard exponential
# Plot
python -c "import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; 
data = np.load('results/phase_diagram/phase_diagram_exponential.npz'); 
res = data['rmse']; J=data['J']; c=data['c']; 
plt.figure(); sns.heatmap(res, xticklabels=np.round(c,2), yticklabels=np.round(J,2), cmap='viridis'); 
plt.xlabel('Shared Noise (c)'); plt.ylabel('Coupling (J)'); plt.title('Phase Diagram (Exponential)'); 
plt.savefig('results/figures/Fig_A9_Phase_Exponential.png')"

# 3. Universality (Fig A10) - Softplus
echo "[3/4] Running Phase Diagram (Softplus)..."
python src/run_phase_diagram.py --config configs/base.yaml --out_dir results/phase_diagram --hazard softplus
# Plot
python -c "import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; 
data = np.load('results/phase_diagram/phase_diagram_softplus.npz'); 
res = data['rmse']; J=data['J']; c=data['c']; 
plt.figure(); sns.heatmap(res, xticklabels=np.round(c,2), yticklabels=np.round(J,2), cmap='viridis'); 
plt.xlabel('Shared Noise (c)'); plt.ylabel('Coupling (J)'); plt.title('Phase Diagram (Softplus)'); 
plt.savefig('results/figures/Fig_A10_Universality_Softplus.png')"

# 4. Structural Equivalence (Fig A11)
echo "[4/4] Running FP vs Age-PDE Comparison..."
python src/run_fp_comparison.py --config configs/base.yaml --out_dir results/figures

echo "Phase II Complete."
