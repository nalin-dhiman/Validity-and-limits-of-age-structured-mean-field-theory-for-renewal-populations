#!/bin/bash
set -e

echo "--- Generating Phase III Traces ---"

# 1. Uncoupled (J=0, c=0) - Fig A4
# Run Sim with Online PDE
# Duration 20s
python src/simulate_population.py --config configs/base.yaml --outfile results/trace_uncoupled.npz --online-pde

# Plot
python -c "import numpy as np; import matplotlib.pyplot as plt; from metrics import ActivityMetrics;
data = np.load('results/trace_uncoupled.npz');
A_mc = data['A']; A_pde = data['A_pde'];
dt = data['config'].item()['defaults']['dt'];
time = np.arange(len(A_mc)) * dt;
# Smooth
A_mc_sm = ActivityMetrics.smooth_trace(A_mc, dt, window_size=0.05);
A_pde_sm = ActivityMetrics.smooth_trace(A_pde, dt, window_size=0.05);
plt.figure(figsize=(10,5));
plt.plot(time, A_mc_sm, label='MC (Smoothed)', alpha=0.8);
plt.plot(time, A_pde_sm, '--', label='Online PDE', linewidth=2);
plt.title('Uncoupled Dynamics (Data-Driven Hazard)');
plt.xlabel('Time (s)'); plt.ylabel('Rate (Hz)'); plt.legend();
plt.savefig('results/figures/Fig_A4_Uncoupled.png');"


# 2. Weak Coupled (J=0.0002, c=0.2) - Fig A5
# Create temp config
python -c "import yaml; d=yaml.safe_load(open('configs/base.yaml')); d['coupling']['J']=0.0002; d['population']['shared_noise_fraction']=0.2; yaml.dump(d, open('configs/coupled.yaml','w'))"

python src/simulate_population.py --config configs/coupled.yaml --outfile results/trace_coupled.npz --online-pde

# Plot
python -c "import numpy as np; import matplotlib.pyplot as plt; from metrics import ActivityMetrics;
data = np.load('results/trace_coupled.npz');
A_mc = data['A']; A_pde = data['A_pde'];
dt = data['config'].item()['defaults']['dt'];
time = np.arange(len(A_mc)) * dt;
A_mc_sm = ActivityMetrics.smooth_trace(A_mc, dt, window_size=0.05);
A_pde_sm = ActivityMetrics.smooth_trace(A_pde, dt, window_size=0.05);
plt.figure(figsize=(10,5));
plt.plot(time, A_mc_sm, label='MC (Smoothed)', alpha=0.8);
plt.plot(time, A_pde_sm, '--', label='Online PDE', linewidth=2);
plt.title('Weak Coupled Dynamics (J=2e-4, c=0.2)');
plt.xlabel('Time (s)'); plt.ylabel('Rate (Hz)'); plt.legend();
plt.savefig('results/figures/Fig_A5_WeakCoupled.png');"

echo "Traces Generated."
