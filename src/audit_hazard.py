import numpy as np
import yaml
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from simulate_population import run_population_sim
from utils import load_config, ensure_dir

def audit_hazard():
    out_dir = 'results/diagnostics/hazard_audit'
    ensure_dir(out_dir)
    
    cfg_path = 'configs/base.yaml'
    cfg = load_config(cfg_path)
    
    N = 5000
    cfg['population']['N'] = N
    cfg['duration'] = 1.0 # Short
    cfg['defaults']['dt'] = 0.0001
    
    # We need to access PDE internals.
    # run_population_sim doesn't return PDE object, but it saves 'rho_field' if we enable online-pde?
    # Actually simulate_population.py saves `rho_field` (snapshots).
    # But it does NOT save `rho_used` from PDE.
    # We instrumented `AgePDESolver` to have `rho_last_used`.
    # But `run_population_sim` creates the solver locally.
    # We must modify `simulate_population.py` to return the solver or save `rho_last_used`?
    # Or we construct the simulation here explicitly (copy logic).
    
    # Alternative: Subclass or Copy logic.
    # Copying logic is safer for audit.
    
    from effective_neuron import EffectiveNeuron
    from age_pde import AgePDESolver
    # from filtering import HazardInferenceAgent # If exists? No, logic inline.
    from scipy.ndimage import gaussian_filter1d
    
    print("Running Hazard Audit Sim...")
    dt = cfg['defaults']['dt']
    T = cfg['duration']
    steps = int(T/dt)
    
    neuron = EffectiveNeuron(cfg, N=N)
    
    R_max = cfg['pde']['R_max']
    dr = cfg['pde']['dr']
    pde = AgePDESolver(T, dt, R_max, dr, cfg['spike_gen']['refractory'])
    
    neuron.last_spike_time = -np.random.uniform(0, 0.1, N)
    
    # Audit Logs
    max_diffs = []
    
    # Snapshots
    rho_hat_snap = []
    rho_used_snap = []
    times = []
    
    # Init PDE (Gaussian)
    ages = -neuron.last_spike_time
    # ... (Binning logic same as simulate_population)
    
    current_time = 0.0
    u_ext = 0.0
    
    for i in range(steps):
        t = i * dt
        
        # 1. Update Ages
        ages = t - neuron.last_spike_time
        
        # 2. Estimate Hazard (MC)
        # Binning logic
        bin_idx = (ages / dr).astype(int)
        bin_idx = np.clip(bin_idx, 0, pde.M)
        n_risk = np.bincount(bin_idx, minlength=pde.M+1)
        
        spikes, _ = neuron.step(t, dt, u_ext)
        
        spk_ages = ages[spikes]
        if len(spk_ages)>0:
            spk_idx = (spk_ages/dr).astype(int)
            spk_idx = np.clip(spk_idx, 0, pde.M)
            n_spk = np.bincount(spk_idx, minlength=pde.M+1)
        else:
            n_spk = np.zeros_like(n_risk)
            
        # Inference
        a0, b0 = 1.0, 0.05
        rho_est = (a0 + n_spk) / (b0 + n_risk * dt)
        
        # Refractory
        mask_ref = pde.r_grid < (pde.ref_period + dr)
        rho_est[mask_ref] = 0.0
        
        # Smoothing
        rho_smooth = gaussian_filter1d(rho_est, sigma=2.0)
        rho_smooth = np.clip(rho_smooth, 0.0, 200.0)
        
        # 3. Step PDE
        pde.step(rho_smooth)
        
        # 4. Audit
        rho_used = pde.rho_last_used
        
        diff = np.abs(rho_smooth - rho_used)
        max_diff = np.max(diff)
        max_diffs.append(max_diff)
        
        if i % 100 == 0:
            rho_hat_snap.append(rho_smooth.copy())
            rho_used_snap.append(rho_used.copy())
            times.append(t)
            
    print(f"Max Hazard Discrepancy: {np.max(max_diffs):.6e} Hz")
    
    # Save CSV
    pd.DataFrame({'t': np.arange(steps)*dt, 'max_diff': max_diffs}).to_csv(os.path.join(out_dir, 'hazard_diff.csv'), index=False)
    
    # Plot Snapshots
    idx = len(times) // 2
    plt.figure(figsize=(10, 5))
    plt.plot(pde.r_grid, rho_hat_snap[idx], label='Rho Hat (MC)')
    plt.plot(pde.r_grid, rho_used_snap[idx], '--', label='Rho Used (PDE)')
    plt.plot(pde.r_grid, np.abs(rho_hat_snap[idx]-rho_used_snap[idx]), 'r:', label='Diff')
    plt.legend()
    plt.title(f'Hazard Audit t={times[idx]:.2f}s')
    plt.xlabel('Age')
    plt.savefig(os.path.join(out_dir, 'hazard_audit_snap.png'))
    print("Saved plot.")

if __name__ == "__main__":
    audit_hazard()
