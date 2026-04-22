import os
import sys

# Ensure headless plotting for server
import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import pandas as pd

import yaml
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from run_phase_scan import run_point, Status
from utils import load_config, ensure_dir

# Import after setting backend and path
def run_diagnostics():
    print(">>> Starting Phase IX Diagnostic Protocol (Server Mode) <<<")
    
    # 1. Config
    base_cfg_path = 'configs/base.yaml'
    out_dir = 'results/phaseIX_diagnostic'
    ensure_dir(out_dir)
    
    if not os.path.exists(base_cfg_path):
        # Fallback for running from scripts dir
        base_cfg_path = '../configs/base.yaml'
        if not os.path.exists(base_cfg_path):
            print("Error: configs/base.yaml not found.")
            sys.exit(1)

    cfg = load_config(base_cfg_path)
    
    # Force R_max logic (Step 3) - CRITICAL for this diagnostic
    R_max_forced = 300.0
    cfg['pde']['R_max'] = R_max_forced 
    print(f"[Diagnostic] Forcing R_max = {R_max_forced}")

    # 2. Parameters (Step 2)
    # Define the 6 points x 2 modes
    points = [
        {'c': 0.0, 'J': 0.0},
        {'c': 0.0, 'J': 0.2},
        {'c': 0.0, 'J': 0.6},
        {'c': 0.5, 'J': 0.0},
        {'c': 0.5, 'J': 0.2},
        {'c': 0.5, 'J': 0.6},
    ]
    
    modes = ['PDE_ONLY', 'ONLINE_PDE']
    results = []
    
    print(f"[Diagnostic] Total Points: {len(points)}. Total Runs: {len(points)*2}")
    
    for pt in points:
        c = pt['c']
        J = pt['J']
        seed = 42 # Fixed seed for diagnostic consistency
        
        for mode in modes:
            print(f"\n--- Running J={J} c={c} Mode={mode} ---")
            
            # Create fresh config for this run
            run_cfg = yaml.safe_load(yaml.dump(cfg)) 
            
            if mode == 'PDE_ONLY':
                run_cfg['pde_only'] = True
            else:
                run_cfg['pde_only'] = False
            
            # Tag the output file to distinguish modes
            run_cfg['robustness_tag'] = mode.lower()
            
            # Execute Simulation
            try:
                res = run_point((run_cfg, J, c, seed, out_dir))
            except Exception as e:
                print(f"!!! EXCEPTION executing point: {e}")
                res = {'status': 'EXEC_ERROR', 'reason': str(e)}
            
            # Collect Metrics
            # If breakdown caught by run_point, metrics might be NaN, which is expected.
            entry = {
                'J': J, 
                'c': c, 
                'seed': seed,
                'mode': mode,
                'status': res.get('status', 'UNKNOWN'),
                'mass_min': res.get('mass_min', np.nan),
                'mass_final': res.get('mass_final', np.nan),
                'tail_mass_max': res.get('tail_mass_max', np.nan),
                't_fail': res.get('t_fail', np.nan),
                'reason': res.get('reason', '')
            }
            results.append(entry)
            
            if res.get('status') != Status.OK:
                print(f"!!! FAILURE: {res.get('status')} Reason: {res.get('reason')}")
            else:
                print(f"OK. MassMin={entry['mass_min']:.4f} TailMax={entry['tail_mass_max']:.4e}")

    # 3. Report Generation
    df = pd.DataFrame(results)
    
    # Save Raw CSV first
    out_csv = os.path.join(out_dir, 'diagnostic_table.csv')
    df.to_csv(out_csv, index=False)
    
    # Print Table
    print("\n" + "="*60)
    print("DIAGNOSTIC RESULTS TABLE")
    print("="*60)
    # Use pandas to string for table-like output without extra deps
    print(df[['J', 'c', 'mode', 'status', 'mass_min', 'tail_mass_max', 'reason']].to_string(index=False, float_format="%.4f"))
    print("="*60)
    
    # 4. Assessment (Step 6)
    assess_verdict(df, out_dir)

def assess_verdict(df, out_dir):
    print("\n>>> AUTOMATED ASSESSMENT <<<")
    
    # Filter
    pde_only = df[df['mode'] == 'PDE_ONLY']
    online = df[df['mode'] == 'ONLINE_PDE']
    
    if pde_only.empty or online.empty:
        print("Insufficient data for assessment.")
        return

    # Helper to check stability
    def is_run_stable(row):
        # OK is stable.
        if row['status'] == Status.OK: return True
        # Allow very minor soft mismatches if mass is good
        if row['status'] in [Status.MISMATCH_TRUNCATION, Status.SOFT_MASS_LEAK]:
             # If mass > 0.99 and tail < 1e-3, we consider it essentially stable for this binary decision
             m = row['mass_min']
             t = row['tail_mass_max']
             # Handle NaNs
             if pd.isna(m) or pd.isna(t): return False
             if m > 0.99 and t < 1e-3: return True
        return False

    # Check stability of all points in each set
    # We are Strict: All 6 points should be stable to be "Stable"
    pde_stable = all(is_run_stable(row) for _, row in pde_only.iterrows())
    online_stable = all(is_run_stable(row) for _, row in online.iterrows())
    
    print(f"PDE_ONLY Set Stable?  {pde_stable}")
    print(f"ONLINE_PDE Set Stable? {online_stable}")
    
    verdict_file = os.path.join(out_dir, 'diagnostic_verdict.txt')
    
    with open(verdict_file, 'w') as f:
        f.write("PHASE IX DIAGNOSTIC VERDICT\n")
        f.write("===========================\n\n")
        
        if pde_stable and not online_stable:
            msg = "VERDICT: CASE A (Coupling Interface Bug)\n" \
                  "Evidence: PDE-only runs are stable (well-posed), but Online-PDE runs fail.\n" \
                  "Action: Focus on fixing the online MC-PDE coupling interface."
            print("\n" + msg)
            f.write(msg)
            
        elif not pde_stable:
             msg = "VERDICT: CASE B (Truncation / Well-Posedness Limitation)\n" \
                   "Evidence: PDE-only runs also lose mass or fail even with R_max=300.\n" \
                   "Action: Reframe Phase Scan as a well-posedness map. Consider tail-closure."
             print("\n" + msg)
             f.write(msg)
             
        elif pde_stable and online_stable:
             msg = "VERDICT: CASE C (Both Stable)\n" \
                   "Evidence: Both modes are stable with R_max=300.\n" \
                   "Action: Previous failures were likely due to R_max < 300. Rerun phase scan with R_max forced."
             print("\n" + msg)
             f.write(msg)
        else:
             msg = "VERDICT: UNDEFINED (Mixed Results)\n" \
                   "Evidence: Online is stable but PDE is not? (Impossible theoretically unless noise artifacts)."
             print("\n" + msg)
             f.write(msg)

    print(f"\nVerdict written to {verdict_file}")

if __name__ == "__main__":
    run_diagnostics()
