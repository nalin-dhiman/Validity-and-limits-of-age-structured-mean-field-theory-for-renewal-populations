#!/bin/bash
set -e

echo ">>> VALIDATION STEP 1: Minimal PDE Run (Mass Conservation) <<<"
# Run short PDE and check diagnostics
python src/simulate_pde.py --config configs/weak.yaml --outfile results/test_pde_fix.npz
# Check mass in the CSV
python -c "
import pandas as pd
df = pd.read_csv('results/test_pde_fix_diagnostics.csv')
min_mass = df['mass'].min()
tail_max = df['q_tail'].max()
print(f'Min Mass: {min_mass}, Max Tail: {tail_max}')
if min_mass < 0.995: 
    print('FAIL: Mass collapsed')
    exit(1)
if tail_max > 1e-6:
    print('FAIL: Tail leak')
    exit(1)
print('PASS: Conservation')
"

echo ">>> VALIDATION STEP 2: R_max Guard Check <<<"
# Temporarily modify config to be bad
cp configs/weak.yaml configs/bad.yaml
sed -i 's/R_max: 65.0/R_max: 5.0/' configs/bad.yaml
set +e
python src/simulate_pde.py --config configs/bad.yaml --outfile results/fail_test.npz > results/fail_log.txt 2>&1
RET=$?
set -e
if [ $RET -eq 0 ]; then
    echo "FAIL: Should have failed with small R_max"
    exit 1
else
    echo "PASS: Correctly rejected small R_max"
fi
rm configs/bad.yaml

echo ">>> VALIDATION STEP 3: Scaling Metric Check (Dry Run) <<<"
# Run tiny scaling
python src/run_scaling.py --config configs/weak.yaml --out_dir results/test_scaling

echo ">>> ALL SYSTEMS GO. Fixes Verified. <<<"
