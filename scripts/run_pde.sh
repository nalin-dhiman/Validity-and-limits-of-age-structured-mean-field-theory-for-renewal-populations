#!/bin/bash
mkdir -p results/logs
python src/simulate_pde.py --config configs/base.yaml --outfile results/logs/pde_baseline.npz
