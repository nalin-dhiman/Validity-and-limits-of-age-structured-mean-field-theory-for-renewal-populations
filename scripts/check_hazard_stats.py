import numpy as np
import yaml
import os
import sys
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from effective_neuron import EffectiveNeuron
from coupling import MeanFieldCoupling
from utils import load_config, set_seed

def check_hazard_stats():
    config_path = 'configs/base.yaml'
    cfg = load_config(config_path)
    
    N = 1000
    T = 10.0
    dt = cfg['defaults']['dt']
    steps = int(T / dt)
    
    # PDE Params
    R_max = cfg['pde']['R_max']
    dr = cfg['pde']['dr']
    ref_period = cfg['spike_gen']['refractory']
    
    # Grid
    M = int(R_max / dr)
    r_grid = np.linspace(0, R_max, M + 1)
    n_bins_pde = len(r_grid)
    
    neuron = EffectiveNeuron(cfg, N=N)
    
    # State
    # Sync start like in population sim
    neuron.last_spike_time = -np.random.uniform(0, 0.1, N)
    
    rhos_max = []
    rhos_min = []
    rhos_mean = []
    
    print(f"Running Hazard Check (N={N}, T={T}s)...")
    
    u_ext = 0.0 # Baseline
    
    for i in range(steps):
        t = i * dt
        
        # 1. Update Ages
        ages = t - neuron.last_spike_time
        
        # 2. Risk Count
        bin_idx = (ages / dr).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins_pde-1)
        n_risk = np.bincount(bin_idx, minlength=n_bins_pde)
        
        # 3. Step Neuron
        spikes, _ = neuron.step(t, dt, u_ext)
        
        # 4. Spike Count
        spking_ages = ages[spikes]
        if len(spking_ages) > 0:
            spk_idx = (spking_ages / dr).astype(int)
            spk_idx = np.clip(spk_idx, 0, n_bins_pde-1)
            n_spk = np.bincount(spk_idx, minlength=n_bins_pde)
        else:
            n_spk = np.zeros(n_bins_pde)
            
        # 5. Inference
        a0 = 1.0
        b0 = 0.05
        rho_est = (a0 + n_spk) / (b0 + n_risk * dt)
        
        # Refractory Mask
        # bins corresponding to r < ref
        # r_grid[j] < ref + dr
        mask_ref = r_grid < (ref_period + dr)
        rho_est[mask_ref] = 0.0
        
        # Smoothing
        rho_smooth = gaussian_filter1d(rho_est, sigma=2.0)
        
        # Stats
        raw_max = np.max(rho_smooth)
        
        if i % 100 == 0:
             rhos_max.append(raw_max)
             rhos_min.append(np.min(rho_smooth))
             rhos_mean.append(np.mean(rho_smooth))
             
    # Stats Summary
    max_h = np.max(rhos_max)
    mean_h = np.mean(rhos_mean)
    min_h = np.min(rhos_min)
    
    print(f"Hazard Statistics (over {T}s):")
    print(f"Max Inferred Hazard (Raw Smooth): {max_h:.2f} Hz")
    print(f"Mean Inferred Hazard: {mean_h:.2f} Hz")
    print(f"Min Inferred Hazard: {min_h:.2f} Hz")
    
    if max_h > 200.0:
        print("Note: Hazard exceeded 200Hz (Clipped in production).")
    else:
        print("Note: Hazard stayed within 200Hz limit.")
        
    # Save CSV
    df = pd.DataFrame({'max': rhos_max, 'mean': rhos_mean, 'min': rhos_min})
    df.to_csv('results/hazard_stats.csv', index=False)
    print("Saved results/hazard_stats.csv")
    
if __name__ == "__main__":
    try:
        check_hazard_stats()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
