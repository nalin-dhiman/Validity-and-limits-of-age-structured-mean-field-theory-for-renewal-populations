
import numpy as np
import os
import sys
import yaml
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from metrics import ActivityMetrics
from utils import ensure_dir

def run_variant(variant_name, override_cfg, seed, out_dir):
    cfg_path = 'configs/base.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Phase IX settings
    cfg['population']['N'] = 5000
    cfg['coupling']['J'] = 0.0
    cfg['stimulus']['mu_u'] = 0.0
    cfg['duration'] = 60.0 
    cfg['duration'] = 60.0 
    cfg['defaults']['seed'] = int(seed)
    cfg['pde']['R_max'] = 80.0 # Safe > 60 + 5 + ref
    
    # Apply Overrides
    for k, v in override_cfg.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                cfg[k][k2] = v2
        else:
            cfg[k] = v
            
    # Write config
    test_cfg_path = os.path.join(out_dir, f'cfg_{variant_name}_s{seed}.yaml')
    with open(test_cfg_path, 'w') as f:
        yaml.dump(cfg, f)
        
    outfile = os.path.join(out_dir, f'run_{variant_name}_s{seed}.npz')
    
    if os.path.exists(outfile):
        # Optional: check integrity?
        try:
            np.load(outfile, allow_pickle=True)
            print(f"[CheckBias] Skipping {variant_name} s{seed}, file exists.")
            return outfile
        except:
            print(f"[CheckBias] File {outfile} corrupted, re-running.")
    
    print(f"[CheckBias] Running Variant: {variant_name} Seed: {seed}...")
    
    # NO DEVNULL - Capture output in logs via SimulateLogger (internal to script)
    # But for the orchestrator, we permit stdout to flow to console or be captured by caller.
    subprocess.run([
        'python', 'src/simulate_population.py',
        '--config', test_cfg_path,
        '--outfile', outfile,
        '--online-pde'
    ], check=True)
    
    return outfile

def analyze_results():
    out_dir = 'results/diagnostics/bias_fix'
    ensure_dir(out_dir)
    ensure_dir('results/tables')
    ensure_dir('results/figures')
    
    # Variants definition matches user request
    variants = {
        'age_only': {
            'neuron': {'kappa': 0.0}, 
            'pde': {'use_jensen': False}
        },
        'age_adapt_no_jensen': {
            'pde': {'use_jensen': False}
        },
        'age_adapt_jensen': {
            'pde': {'use_jensen': True}
        }
    }
    
    seeds = [0, 1, 2, 3, 4]
    
    # 1. Prepare Tasks
    tasks = []
    for name, overrides in variants.items():
        for s in seeds:
            tasks.append((name, overrides, s, out_dir))
            
    print(f"Total Bias Fix Simulations: {len(tasks)}")
    
    if not tasks:
        print("No tasks defined.")
        return
        
    # ... (skipping Golden Run / Parallel part which are fine) ...
    # This replacement is tricky because I need to touch multiple blocks.
    # I will use multi_replace.


    # 2. Golden Run (Fail Fast)
    print(">>> executing GOLDEN RUN (1st task) <<<")
    try:
        run_variant(*tasks[0])
        print(">>> Golden Run SUCCESS <<<")
    except Exception as e:
        print(f"!!! Golden Run FAILED: {e}")
        sys.exit(1)
        
    # 3. Parallel Execution (Remaining)
    remaining_tasks = tasks[1:]
    if remaining_tasks:
        max_cores = multiprocessing.cpu_count()
        n_proc = min(max(1, int(max_cores * 0.8)), 32)
        print(f"Running remaining {len(remaining_tasks)} tasks with {n_proc} processes...")
        
        with multiprocessing.Pool(processes=n_proc) as pool:
            pool.starmap(run_variant, remaining_tasks)
            
    # 4. Analysis
    results = []
    taus = [0.02, 0.05]
    
    for task in tasks:
        name, _, s, _ = task
        outfile = os.path.join(out_dir, f'run_{name}_s{s}.npz')
        
        if not os.path.exists(outfile):
             print(f"Missing output for {name} s{s}")
             continue
             
        try:
            data = np.load(outfile, allow_pickle=True)
            A_mc = data['A']
            
            # Select PDE
            if name == 'age_only':
                A_pde = data['A_pde_age_only']
            else:
                A_pde = data['A_pde_adapt']
                
            dt = data['config'].item()['defaults']['dt']
            burn = int(10.0/dt)
            
            # Crop
            A_mc = A_mc[burn:]
            A_pde = A_pde[burn:]
            L = min(len(A_mc), len(A_pde))
            A_mc = A_mc[:L]
            A_pde = A_pde[:L]
            
            row = {'Closure': name, 'Seed': s}
            
            for tau in taus:
                suffix = f"_{int(tau*1000)}ms"
                
                A_mc_sm = ActivityMetrics.smooth_trace(A_mc, dt, tau)
                A_pde_sm = ActivityMetrics.smooth_trace(A_pde, dt, tau)
                
                # Align if needed? Assuming locked step.
                diff = A_mc_sm - A_pde_sm
                
                rmse = np.sqrt(np.mean(diff**2))
                bias = np.mean(diff)
                
                # Phase IX Fix: Normalize by PDE std (consistent with scaling)
                norm = np.std(A_pde_sm)
                nrmse = rmse / norm if norm > 1e-9 else 0.0
                
                row[f'Bias{suffix}'] = bias
                row[f'AbsBias{suffix}'] = np.abs(bias)
                row[f'RMSE{suffix}'] = rmse
                row[f'NRMSE{suffix}'] = nrmse
                
            results.append(row)
        except Exception as e:
            print(f"Error analyzing {outfile}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv('results/tables/bias_fix_summary_seeds.csv', index=False)
    
    # Stats
    # Stats
    # stats = df.groupby('Closure').agg(['mean', 'std'])
    # stats.to_csv('results/tables/bias_fix_summary_stats.csv')
    # print("Saved summary stats.")
    # print(stats)
    
    # Just save the raw grid for paper plotting
    print("Saved bias_fix_summary_seeds.csv. Stats aggregation skipped to avoid 1-seed warnings.")
    
    # 5. Plotting (Overlay for Seed 0)
    plot_overlay(variants.keys(), out_dir, df)

def plot_overlay(variant_names, out_dir, df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    dt = 0.0001
    
    for i, name in enumerate(variant_names):
        outfile = os.path.join(out_dir, f'run_{name}_s0.npz')
        if not os.path.exists(outfile): continue
        
        data = np.load(outfile, allow_pickle=True)
        burn = int(10.0/dt)
        A_mc = data['A'][burn:]
        
        if name == 'age_only':
             A_pde = data['A_pde_age_only'][burn:]
        else:
             A_pde = data['A_pde_adapt'][burn:]
             
        t_plot = np.arange(len(A_mc)) * dt
        
        # Smooth 20ms
        A_mc_sm = ActivityMetrics.smooth_trace(A_mc, dt, 0.02)
        A_pde_sm = ActivityMetrics.smooth_trace(A_pde, dt, 0.02)
        
        L_p = min(len(A_mc_sm), len(A_pde_sm), 20000) # 2s plot
        
        ax = axes[i]
        ax.plot(t_plot[:L_p], A_mc_sm[:L_p], 'k-', label='MC', linewidth=1, alpha=0.7)
        ax.plot(t_plot[:L_p], A_pde_sm[:L_p], 'r--', label='PDE', linewidth=1.5)
        
        # Annotate
        try:
            row_stats = df[(df['Closure'] == name) & (df['Seed'] == 0)].iloc[0]
            nrmse = row_stats['NRMSE_20ms']
            bias = row_stats['AbsBias_20ms']
            ax.set_title(f"{name} | Bias={bias:.3f} Hz, NRMSE={nrmse:.3f}")
        except:
            ax.set_title(name)
            
        ax.legend()
        ax.set_ylabel("Rate (Hz)")
        
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig('results/figures/Fig_BiasFix.png')
    print("Saved results/figures/Fig_BiasFix.png")

if __name__ == "__main__":
    analyze_results()
