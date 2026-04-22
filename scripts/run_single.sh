#!/bin/bash
mkdir -p results/figures
python src/simulate_single.py --config configs/base.yaml --out_dir results/figures
