#!/bin/bash
mkdir -p results/logs
python src/simulate_population.py --config configs/base.yaml --outfile results/logs/pop_baseline.npz
