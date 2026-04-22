import sys
import os
import argparse
import numpy as np
import yaml
import subprocess
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import load_config, ensure_dir

def run_phase_diagram(config_path, out_dir, hazard_type='exponential'):
    ensure_dir(out_dir)
    cfg = load_config(config_path)
    
    # Set hazard type if specified
    if 'spike_gen' not in cfg: cfg['spike_gen'] = {}
    cfg['spike_gen']['type'] = hazard_type
    
    # Param grids - Refined for "Sharp" boundary
    J_vals = np.linspace(0, 50, 11) # Broader range to find J_max around strong coupling
    c_vals = np.linspace(0, 1.0, 11)
    
    results = np.zeros((len(J_vals), len(c_vals)))
    
    base_cfg_path = os.path.join(out_dir, f'config_phase_{hazard_type}.yaml')
    with open(base_cfg_path, 'w') as f:
        yaml.dump(cfg, f)
        
    for i, J in enumerate(J_vals):
        for k, c in enumerate(c_vals):
            # Create Run Config
            run_cfg = cfg.copy()
            run_cfg['coupling']['J'] = float(J)
            run_cfg['population']['shared_noise_fraction'] = float(c)
            # Use smaller N for speed if mapping phase diagram?
            # Prompt says "Repeat for at least two values of N".
            # Let's stick to N=2000 for the main map.
            N = 2000
            run_cfg['population']['N'] = N
            
            tmp_cfg_path = os.path.join(out_dir, f'run_{hazard_type}_J{i}_c{k}.yaml')
            with open(tmp_cfg_path, 'w') as f:
                yaml.dump(run_cfg, f)
            
            mc_out = os.path.join(out_dir, f'mc_{hazard_type}_J{i}_c{k}.npz')
            pde_out = os.path.join(out_dir, f'pde_{hazard_type}_J{i}_c{k}.npz')
            
            # Run
            # Using subprocess to ensure clean state
            subprocess.run(['python', 'src/simulate_population.py', '--config', tmp_cfg_path, '--outfile', mc_out], check=True)
            subprocess.run(['python', 'src/simulate_pde.py', '--config', tmp_cfg_path, '--outfile', pde_out], check=True)
            
            # Compute RMSE
            mc_data = np.load(mc_out)
            pde_data = np.load(pde_out)
            
            A_mc = mc_data['A']
            A_pde = pde_data['A']
            
            factor = len(A_pde) // len(A_mc)
            if factor > 0:
                A_pde_binned = A_pde[:len(A_mc)*factor].reshape(-1, factor).mean(axis=1)
                rmse = np.sqrt(np.mean((A_mc - A_pde_binned)**2))
            else:
                rmse = np.sqrt(np.mean((A_mc - A_pde)**2))
                
            results[i, k] = rmse
            
            # Clean up heavy files
            os.remove(mc_out)
            os.remove(pde_out)
            os.remove(tmp_cfg_path)
            
    # Save Grid
    np.savez(os.path.join(out_dir, f'phase_diagram_{hazard_type}.npz'), rmse=results, J=J_vals, c=c_vals)
    df = pd.DataFrame(results, index=[f"J={x:.2f}" for x in J_vals], columns=[f"c={x:.2f}" for x in c_vals])
    df.to_csv(os.path.join(out_dir, f'phase_diagram_{hazard_type}.csv'))
    
    return results, J_vals, c_vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--out_dir', default='results/phase_diagram')
    parser.add_argument('--hazard', default='exponential')
    args = parser.parse_args()
    
    run_phase_diagram(args.config, args.out_dir, args.hazard)
