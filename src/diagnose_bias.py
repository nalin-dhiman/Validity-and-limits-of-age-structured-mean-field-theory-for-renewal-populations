import numpy as np
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from metrics import ActivityMetrics
from utils import ensure_dir

def run_sim_task(args):
    N, seed, out_dir, cfg_template_path = args
    
    # Load config template
    with open(cfg_template_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['defaults']['seed'] = seed
    cfg['population']['N'] = N
    # Uncoupled, Constant Input
    cfg['coupling']['J'] = 0.0
    cfg['stimulus']['mu_u'] = 0.0 # Baseline
    cfg['stimulus']['type'] = 'ou' # Keep OU but with small sigma, or constant? user said mu=0. 
    # Ensure T=60 -> Debug T=1.0
    cfg['duration'] = 1.0
    
    # Run
    # Write temp config
    cfg_path = os.path.join(out_dir, f'cfg_N{N}_s{seed}.yaml')
    out_file = os.path.join(out_dir, f'sim_N{N}_s{seed}.npz')
    
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f)
        
    # Call script (safer than import for memory)
    cmd = [
        'python', 'src/simulate_population.py',
        '--config', cfg_path,
        '--outfile', out_file,
        '--online-pde'
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Return path
    return out_file

def diagnose_bias():
    out_dir = 'results/diagnostics/bias_variance'
    ensure_dir(out_dir)
    
    cfg_base = 'configs/base.yaml'
    N_vals = [500, 1000, 2000, 5000, 10000]
    K = 10
    
    dt = 0.0001
    burn_in = 10.0
    burn_steps = int(burn_in / dt)
    tau = 0.02 # 20ms smoothing
    
    tasks = []
    seeds = range(42, 42+K)
    
    for N in N_vals:
        for s in seeds:
            tasks.append((N, s, out_dir, cfg_base))
            
    print(f"Launching {len(tasks)} simulations (N={N_vals}, K={K}) sequentially...")
    
    # Run Sequential
    sim_outputs = {} # Key (N, s) -> file
    
    total_tasks = len(tasks)
    for i, task in enumerate(tasks):
        N_curr, s_curr, _, _ = task
        print(f"Running Task {i+1}/{total_tasks}: N={N_curr}, seed={s_curr}...")
        try:
            outfile = run_sim_task(task)
            sim_outputs[(N_curr, s_curr)] = outfile
        except Exception as e:
            print(f"Task Failed: {e}")
            
    # Map back (Sequential logic already filled sim_outputs)
        
    # Analysis
    stats = []
    
    print("Analyzing Bias and Noise...")
    for N in N_vals:
        A_mc_stack = []
        A_pde_stack = []
        
        for s in seeds:
            data = np.load(sim_outputs[(N, s)])
            A_mc_raw = data['A'][burn_steps:]
            A_pde_raw = data['A_pde'][burn_steps:]
            
            # Smooth
            A_mc_sm = ActivityMetrics.smooth_trace(A_mc_raw, dt, tau)
            A_pde_sm = ActivityMetrics.smooth_trace(A_pde_raw, dt, tau)
            
            # Crop to min length
            L = min(len(A_mc_sm), len(A_pde_sm))
            A_mc_stack.append(A_mc_sm[:L])
            A_pde_stack.append(A_pde_sm[:L])
            
            # Cleanup
            # os.remove(sim_outputs[(N, s)]) # Keep for now
            
        A_mc_stack = np.array(A_mc_stack) # (K, T_steps)
        A_pde_stack = np.array(A_pde_stack)
        
        # Means
        Abar_MC = np.mean(A_mc_stack, axis=0)
        Abar_PDE = np.mean(A_pde_stack, axis=0) # Mean of PDE responses
        
        # Bias = RMSE(Abar_MC, Abar_PDE)
        diff = Abar_MC - Abar_PDE
        Bias = np.sqrt(np.mean(diff**2))
        
        # Noise = Mean_t(Std_k(A_mc))
        Std_MC_t = np.std(A_mc_stack, axis=0)
        Noise = np.mean(Std_MC_t)
        
        # Also compute Bias relative to Rate (Normalized)
        Rate = np.mean(Abar_MC)
        
        stats.append({
            'N': N,
            'Bias': Bias,
            'Noise': Noise,
            'Rate': Rate,
            'Bias_Rel': Bias/Rate,
            'Noise_Rel': Noise/Rate
        })
        print(f"N={N}: Bias={Bias:.4f}, Noise={Noise:.4f}")
        
    # Save CSV
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(out_dir, 'bias_noise.csv'), index=False)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(df['N'], df['Bias'], 'o-', label='Bias')
    plt.loglog(df['N'], df['Noise'], 's-', label='Noise')
    
    # Plot N^-0.5 reference
    ref_x = np.array(N_vals)
    ref_y = df['Noise'].iloc[0] * (ref_x / ref_x[0])**(-0.5)
    plt.loglog(ref_x, ref_y, 'k--', label='N^-0.5')
    
    plt.xlabel('N')
    plt.ylabel('Magnitude (Hz)')
    plt.legend()
    plt.title('Bias-Variance Decomposition')
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(out_dir, 'bias_noise_scaling.png'))
    print("Saved plot.")

if __name__ == "__main__":
    diagnose_bias()
