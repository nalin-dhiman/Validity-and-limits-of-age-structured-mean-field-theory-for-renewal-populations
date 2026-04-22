import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from effective_neuron import EffectiveNeuron
from coupling import MeanFieldCoupling
from age_pde import AgePDESolver
from utils import load_config, set_seed
from metrics import ActivityMetrics
from scipy.ndimage import gaussian_filter1d

def run_sim_inline(cfg, N, seed, T):
    # Setup
    dt = cfg['defaults']['dt']
    steps = int(T / dt)
    
    # Components
    neuron = EffectiveNeuron(cfg, N=N, seed=seed)
    
    # PDE (Online)
    R_max = cfg['pde']['R_max']
    dr = cfg['pde']['dr']
    pde = AgePDESolver(T, dt, R_max, dr, cfg['spike_gen']['refractory'])
    
    # Init PDE
    ages = -neuron.last_spike_time
    # Binning
    n_bins_pde = len(pde.r_grid) # M+1?
    # Actually pde.r_grid has M+1 points.
    
    A_mc = np.zeros(steps)
    A_pde = np.zeros(steps)
    
    u_ext = 0.0 # Baseline
    
    # Loop
    for i in range(steps):
        t = i * dt
        
        # 1. Update Ages
        ages = t - neuron.last_spike_time
        
        # 2. Estimate Hazard (MC)
        bin_idx = (ages / dr).astype(int)
        bin_idx = np.clip(bin_idx, 0, len(pde.r_grid)-2) # Index into q (0..M)
        n_risk = np.bincount(bin_idx, minlength=len(pde.r_grid))
        
        spikes, _ = neuron.step(t, dt, u_ext)
        
        # Spikes for inference
        spk_ages = ages[spikes]
        if len(spk_ages) > 0:
            spk_idx = (spk_ages / dr).astype(int)
            spk_idx = np.clip(spk_idx, 0, len(pde.r_grid)-2)
            n_spk = np.bincount(spk_idx, minlength=len(pde.r_grid))
        else:
            n_spk = np.zeros(len(pde.r_grid))
            
        # Inference
        a0, b0 = 1.0, 0.05
        # Ensure shapes match
        if len(n_risk) != len(n_spk):
             n_spk = n_spk[:len(n_risk)] # Should match
             
        rho_est = (a0 + n_spk) / (b0 + n_risk * dt)
        
        # Refractory
        mask_ref = pde.r_grid < (pde.ref_period + dr) 
        # pde.r_grid size vs rho_est size?
        # n_risk size is dependent on bin_idx range
        # ensure rho_est matches pde.q size
        if len(rho_est) < len(pde.q):
             pad = len(pde.q) - len(rho_est)
             rho_est = np.pad(rho_est, (0, pad))
        elif len(rho_est) > len(pde.q):
             rho_est = rho_est[:len(pde.q)]
             
        rho_est[mask_ref[:len(rho_est)]] = 0.0
        
        # Smooth
        rho_smooth = gaussian_filter1d(rho_est, sigma=2.0)
        rho_smooth = np.clip(rho_smooth, 0.0, 200.0)
        
        # 3. Step PDE
        A_pde[i] = pde.step(rho_smooth)
        
        # 4. Metric
        A_mc[i] = np.sum(spikes) / (N * dt)
        
    return A_mc, A_pde

def diagnose_bias_inline():
    print("Running Inline Diagnostics (N=500, 10000, K=5)...")
    out_dir = 'results/diagnostics/bias_variance'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    cfg = load_config('configs/base.yaml')
    dt = cfg['defaults']['dt']
    T = 10.0 # Short but sufficient
    burn_steps = int(2.0 / dt)
    
    N_vals = [500, 10000]
    K = 5
    
    stats = []
    
    for N in N_vals:
        print(f" Simulating N={N}...")
        A_mc_stack = []
        A_pde_stack = []
        
        for k in range(K):
            print(f"  Seed {k}...")
            seed = 42 + k
            A_mc, A_pde = run_sim_inline(cfg, N, seed, T)
            
            # Smooth
            A_mc_sm = ActivityMetrics.smooth_trace(A_mc[burn_steps:], dt, 0.02)
            A_pde_sm = ActivityMetrics.smooth_trace(A_pde[burn_steps:], dt, 0.02)
            
            L = min(len(A_mc_sm), len(A_pde_sm))
            A_mc_stack.append(A_mc_sm[:L])
            A_pde_stack.append(A_pde_sm[:L])
            
        A_mc_stack = np.array(A_mc_stack)
        A_pde_stack = np.array(A_pde_stack)
        
        # Metrics
        # Bias: RMSE(Mean(MC), Mean(PDE))
        Abar_MC = np.mean(A_mc_stack, axis=0)
        Abar_PDE = np.mean(A_pde_stack, axis=0)
        
        diff = Abar_MC - Abar_PDE
        Bias = np.sqrt(np.mean(diff**2))
        
        # Noise: Mean(Std(MC))
        Noise = np.mean(np.std(A_mc_stack, axis=0))
        Rate = np.mean(Abar_MC)
        
        stats.append({'N': N, 'Bias': Bias, 'Noise': Noise, 'Rate': Rate})
        print(f"RESULT N={N}: Bias={Bias:.4f}, Noise={Noise:.4f}")
        
    df = pd.DataFrame(stats)
    df.to_csv('results/diagnostics/bias_inline.csv', index=False)
    print("Saved results/diagnostics/bias_inline.csv")

if __name__ == "__main__":
    diagnose_bias_inline()
