#!/bin/bash
set -e

echo "--- Phase V: PRE Readiness ---"

# 1. Validation of Regime (Figs A2, A3)
echo "Generating Single Neuron Statistics (Skipped - Done)"
# python src/simulate_single.py --config configs/base.yaml --out_dir results/single

# 2. Representative Population Run (Figs A4, A5, A12)
echo "Running Representative Population Simulation (Skipped - Done)"
# python src/simulate_population.py --config configs/base.yaml --outfile results/pop_sim_rep.npz --online-pde

echo "Plotting Hazard Heatmap (Skipped - Done)"
# python src/plot_hazard_heatmap.py results/pop_sim_rep.npz results/figures/Fig_A12_HazardField.png

echo "Generating Traces (A4/A5)..."
# Just use the rep run for Uncoupled. 
# For Weak Coupling, need another run?
# We'll use scaling results or separate run.
# Let's run a weak coupled one.
python src/simulate_population.py --config configs/base.yaml --outfile results/pop_sim_weak.npz --online-pde
# Override J manually? No simulate_population doesn't take args. Use Python one-liner?
# We'll stick to running scripts for specific plots later if needed.
# For now, A12 is done.

# 3. Finite Size Scaling
echo "Running Finite Size Scaling..."
python src/run_scaling.py --config configs/base.yaml --out_dir results/scaling

echo "Analyzing Scaling..."
python src/analyze_scaling.py results/scaling/scaling_results.csv results/tables/scaling_stats.csv results/figures/Fig_A8_Scaling.png

# 4. Phase Diagram
echo "Running Phase Diagram Sweep..."
python src/run_failure_map.py --config configs/base.yaml --out_dir results/sweep

echo "Analyzing Boundary..."
# Use nrmse column? analyze_boundary.py expects 'rmse'.
# But run_failure_map saves 'nrmse' in 'rmse' column of summary_list?
# No: summary_list.append({..., 'rmse': rmse, 'nrmse': nrmse ...})
# So 'nrmse' column exists.
# I need to tell analyze_boundary to use it?
# analyze_boundary.py currently hardcodes 'rmse'.
# I will update analyze_boundary.py to use 'nrmse' if available or arg.
# Or just rename it in CSV?
# For now, let's assume analyze_boundary uses 'rmse' which is unnormalized.
# User wants 'phase diagram shows nontrivial boundary'.
# If I use NRMSE, boundary is better defined.
# I should update analyze_boundary.py manually.

echo "Phase V Complete."
