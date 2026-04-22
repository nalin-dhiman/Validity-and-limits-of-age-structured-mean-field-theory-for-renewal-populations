#!/bin/bash
set -e

# Function to update yaml
update_yaml() {
    python -c "import yaml; d=yaml.safe_load(open('$1')); $2; yaml.dump(d, open('$3','w'))"
}

echo "Running Paper A Repairs (Phase I Hardening)..."

mkdir -p results/figures
mkdir -p results/logs
mkdir -p results/sweep
mkdir -p results/tables

# 1. Single Neuron (Figs A1, A2, A3)
echo "Generating Single Neuron Figures (Calibrated)..."
python src/simulate_single.py --config configs/base.yaml --out_dir results/figures

# 2. Comparison: Uncoupled (Fig A4)
echo "Running Uncoupled Comparison (Long T=20s)..."
update_yaml "configs/base.yaml" "d['coupling']['J']=0.0; d['duration']=20.0" "configs/uncoupled.yaml"

python src/simulate_population.py --config configs/uncoupled.yaml --outfile results/logs/mc_uncoupled.npz
python src/simulate_pde.py --config configs/uncoupled.yaml --outfile results/logs/pde_uncoupled.npz

python src/compare_sims.py --mc results/logs/mc_uncoupled.npz --pde results/logs/pde_uncoupled.npz --out results/figures/Fig_A4_Uncoupled.png --title "Uncoupled Activity"

# 3. Comparison: Weak Coupling (Fig A5)
echo "Running Weak Coupling Comparison (Long T=20s, J=2, tau=10ms)..."
update_yaml "configs/base.yaml" "d['coupling']['J']=2.0; d['duration']=20.0" "configs/weak.yaml"

python src/simulate_population.py --config configs/weak.yaml --outfile results/logs/mc_weak.npz
python src/simulate_pde.py --config configs/weak.yaml --outfile results/logs/pde_weak.npz

python src/compare_sims.py --mc results/logs/mc_weak.npz --pde results/logs/pde_weak.npz --out results/figures/Fig_A5_WeakCoupled.png --title "Weak Coupling (J=2)"

# 4. Diagnostics (Fig A6) - Check Mass Leaks
echo "Plotting Diagnostics..."
python src/compare_sims.py --pde results/logs/pde_weak.npz --type diagnostics --out results/figures/Fig_A6_Diagnostics.png
# Save diag table
cp results/logs/pde_weak_diagnostics.csv results/tables/pde_diagnostics.csv

# 5. Failure Map (Fig A7)
echo "Generating Failure Map (Detailed)..."
python src/run_failure_map.py --config configs/base.yaml --out_dir results/sweep
cp results/sweep/Fig_A7_FailureMap.png results/figures/Fig_A7_FailureMap.png

echo "Done! Check results/figures/ and results/tables/"
