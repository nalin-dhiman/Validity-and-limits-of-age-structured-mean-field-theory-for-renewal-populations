#!/bin/bash
set -e

echo "--- Phase VI: Validation ---"

# 1. Single Neuron (F-I, ISI)
echo "Generating Single Neuron Statistics..."
python src/simulate_single.py --config configs/base.yaml --out_dir results/single

# 2. Population (Mass Check)
echo "Running Population Check..."
python src/simulate_population.py --config configs/base.yaml --outfile results/pop_sim_check.npz --online-pde

echo "Validation Complete. Check results/single/ and results/pop_sim_check_mass_debug.csv"
