import sys
import os
import argparse
import numpy as np
import yaml
import subprocess
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import load_config, ensure_dir
import pandas as pd

from metrics import ActivityMetrics

import multiprocessing

def run_single_sweep(args):
    cfg, J, c, out_dir = args
    
    # Create temp config
    run_cfg = cfg.copy()
    run_cfg['coupling']['J'] = float(J)
    run_cfg['population']['shared_noise_fraction'] = float(c)
    run_cfg['population']['N'] = 2000 # Standard
    
    tmp_cfg_path = os.path.join(out_dir, f'config_J{J}_c{c}.yaml')
    with open(tmp_cfg_path, 'w') as f:
        yaml.dump(run_cfg, f)
        
    # Files
    mc_out = os.path.join(out_dir, f'sim_J{J}_c{c}.npz')
    
    # Run (Single command with online-pde)
    subprocess.run(['python', 'src/simulate_population.py', 
                  '--config', tmp_cfg_path, 
                  '--outfile', mc_out, 
                  '--online-pde'], check=True)
                  
    return J, c, mc_out, tmp_cfg_path

def run_sweep(base_config_path, out_dir):
    ensure_dir(out_dir)
    cfg = load_config(base_config_path)
    
     # Calibrated Reform: J must be small for mV scale (sigma=0.003)
    # Target J*A ~ 0.003. If A~20Hz, J ~ 1.5e-4.
    J_vals = [0.0, 0.0001, 0.0002, 0.0005] 
    c_vals = [0.0, 0.2, 0.5, 0.8]
    
    results = np.zeros((len(J_vals), len(c_vals)))
    nrmse_grid = np.zeros((len(J_vals), len(c_vals)))
    var_mc = np.zeros_like(results)
    var_pde = np.zeros_like(results)
    
    summary_list = []
    
    # Increase duration
    cfg['duration'] = 60.0 # T=60s required
    
    tasks = []
    for J in J_vals:
        for c in c_vals:
            tasks.append((cfg, J, c, out_dir))
            
    print(f"Launching {len(tasks)} tasks in parallel...")
    
    # Map (Parallel)
    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
        completed = pool.map(run_single_sweep, tasks)
        
    # Process
    for (J, c, mc_out, tmp_cfg_path) in completed:
        # Match indices
        i = J_vals.index(J)
        k = c_vals.index(c)
        
        # Match
        data = np.load(mc_out)
        A_mc = data['A']
        A_pde = data['A_pde']
        
        dt_sim = cfg['defaults']['dt']
        start_step = int(10.0 / dt_sim)
        
        # NRMSE
        # Truncate if needed
        L = min(len(A_mc), len(A_pde))
        A_mc = A_mc[:L]
        A_pde = A_pde[:L]
        
        nrmse, rmse = ActivityMetrics.compute_nrmse(A_mc[start_step:], A_pde[start_step:], dt_sim, smooth_window=0.05)
        
        results[i, k] = nrmse # Store NRMSE for plotting!
        nrmse_grid[i, k] = nrmse
        
        # Variance check on SMOOTHED
        A_mc_sm = ActivityMetrics.smooth_trace(A_mc[start_step:], dt_sim, 0.05)
        A_pde_sm = ActivityMetrics.smooth_trace(A_pde[start_step:], dt_sim, 0.05)
        
        v_mc = np.var(A_mc_sm)
        v_pde = np.var(A_pde_sm)
        
        var_mc[i, k] = v_mc
        var_pde[i, k] = v_pde
        
        summary_list.append({
            'J': J, 'c': c, 'rmse': rmse, 'nrmse': nrmse, 'var_mc': v_mc, 'var_pde': v_pde
        })
        
        # Wipe large files
        if os.path.exists(mc_out): os.remove(mc_out)
        if os.path.exists(tmp_cfg_path): os.remove(tmp_cfg_path)
            
    # Save Grid
    np.savez(os.path.join(out_dir, 'failure_map.npz'), rmse=results, nrmse=nrmse_grid, J=J_vals, c=c_vals)
    # Save CSV
    out_table_dir = out_dir.replace('sweep', 'tables')
    if not os.path.exists(out_table_dir): os.makedirs(out_table_dir)
    pd.DataFrame(summary_list).to_csv(os.path.join(out_table_dir, 'failure_map.csv'), index=False)
    
    # Plot
    from plots import plot_failure_map2
    plot_failure_map2(nrmse_grid, J_vals, c_vals, os.path.join(out_dir, 'Fig_A7_FailureMap.png'))
    print("Generated Fig A7")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--out_dir', default='results/sweep')
    args = parser.parse_args()
    
    run_sweep(args.config, args.out_dir)
