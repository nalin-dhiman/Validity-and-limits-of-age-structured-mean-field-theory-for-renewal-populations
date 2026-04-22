import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import ensure_dir

def check_variance():
    out_dir = 'results/diagnostics/variance_check'
    ensure_dir(out_dir)
    
    # Run short Sim
    cfg_path = 'configs/base.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['population']['N'] = 50000 # High N for good histograms
    cfg['coupling']['J'] = 0.0
    cfg['duration'] = 5.0 # Enough to settle
    
    test_cfg = os.path.join(out_dir, 'cfg.yaml')
    with open(test_cfg, 'w') as f:
        yaml.dump(cfg, f)
        
    outfile = os.path.join(out_dir, 'run.npz')
    
    print("Running Variance Check Simulation (Golden Run)...")
    # This uses the UPDATED simulate_population.py which saves pop_V
    # Using check=True to enable fast fail
    try:
        subprocess.run([
            'python', 'src/simulate_population.py',
            '--config', test_cfg,
            '--outfile', outfile
            # '--online-pde' # DISABLED for speed (N=50k). Comparison is vs Theory.
        ], check=True)
    except subprocess.CalledProcessError:
        print("CRITICAL: Variance simulation failed.")
        sys.exit(1)
    
    # Analyze
    data = np.load(outfile, allow_pickle=True)
    pop_V = data['pop_V']
    pop_last_spike = data['pop_last_spike_time']
    # If using N=50k, these are arrays of size 50k
    
    t_end = cfg['duration']
    ages = t_end - pop_last_spike
    
    # Bin by age
    dr = 0.005 # 5ms bins
    max_age = 0.5 # 500ms
    bins = np.arange(0, max_age + dr, dr)
    centers = bins[:-1] + dr/2
    
    age_indices = np.digitize(ages, bins) - 1
    
    mc_var = []
    mc_mean = []
    counts = []
    
    for i in range(len(centers)):
        mask = age_indices == i
        if np.sum(mask) > 100:
            v_subset = pop_V[mask]
            mc_var.append(np.var(v_subset))
            mc_mean.append(np.mean(v_subset))
            counts.append(len(v_subset))
        else:
            mc_var.append(np.nan)
            mc_mean.append(np.nan)
            counts.append(0)
            
    mc_var = np.array(mc_var)
    
    # Theoretical Variance
    # sigma^2 * tau_m / 2 * (1 - exp(-2s/tau_m))
    # We assume v(0) is fixed.
    p = cfg['neuron']
    sigma = p['sigma']
    tau_m = p['tau_m']
    
    theory_var = (sigma**2 * tau_m / 2.0) * (1.0 - np.exp(-2.0 * centers / tau_m))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(centers, mc_var, 'ko', label='MC Variance', alpha=0.5)
    plt.plot(centers, theory_var, 'r-', label='Theory (OU Check)', linewidth=2)
    plt.xlabel('Age (s)')
    plt.ylabel('Variance of V')
    plt.title('Validation of Jensen Correction Assumption')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/diagnostics/Fig_Variance.png')
    print("Saved results/diagnostics/Fig_Variance.png")
    
    # Calculate agreement error
    valid = ~np.isnan(mc_var) & (theory_var > 1e-9)
    mse = np.mean((mc_var[valid] - theory_var[valid])**2)
    rmse = np.sqrt(mse)
    
    rel_err = np.abs(mc_var[valid] - theory_var[valid]) / theory_var[valid]
    mean_rel_err = np.mean(rel_err)
    
    print(f"Variance RMSE: {rmse:.6f}")
    print(f"Mean Relative Error: {mean_rel_err:.4f}")
    
    if mean_rel_err > 0.5:
        print("WARNING: large relative error > 50%. Check implementation.")
    else:
        print("Variance check PASSED (O(1) relative error).")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        'age_center': centers,
        'mc_variance': mc_var,
        'theory_variance': theory_var,
        'count': counts
    })
    ensure_dir('results/tables')
    df.to_csv('results/tables/variance_compare.csv', index=False)
    print("Saved results/tables/variance_compare.csv")

if __name__ == "__main__":
    check_variance()
