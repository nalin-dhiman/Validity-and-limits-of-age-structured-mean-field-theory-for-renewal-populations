import numpy as np
import os
import sys
import yaml
import pandas as pd
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from metrics import ActivityMetrics
from utils import ensure_dir

def benchmark_renewal():
    out_dir = 'results/diagnostics/renewal_benchmark'
    ensure_dir(out_dir)
    
    cfg_base = 'configs/base.yaml'
    N_vals = [500, 2000, 10000]
    
    # We need a new simulation script for Renewal Process.
    # Instead of modifying `simulate_population.py`, we write `src/run_renewal_sim.py`.
    # But for brevity, we can just generate MC spikes in this script and run PDE.
    
    from age_pde import AgePDESolver
    # from filtering import HazardInferenceAgent
    
    dt = 0.0001
    T = 60.0
    steps = int(T/dt)
    
    # Define Ground Truth Hazard
    # Hard refractory 5ms, then constant 20Hz.
    ref = 0.005
    rate = 20.0 # Hz
    
    # Run loop
    stats = []
    
    for N in N_vals:
        print(f"Running Renewal Benchmark N={N}...")
        
        # MC Simulation (Gillespie or Time-Step)
        # Time-Step for consistency
        spikes = np.zeros(steps, dtype=int)
        last_spike = -np.random.exponential(1.0/rate, N) # Poisson start? Or Uniform Age
        # Uniform Age [0, 1/rate]
        last_spike = -np.random.uniform(0, 0.1, N)
        
        # Hazard Function
        # rho(age) = rate if age > ref else 0
        
        # We need A(t).
        A_rec = np.zeros(steps)
        
        ages = -last_spike
        
        for i in range(steps):
            t = i * dt
            ages += dt # Evolution
            
            # Hazard
            rho = np.where(ages > ref, rate, 0.0)
            
            # Spike Prob
            p = 1.0 - np.exp(-rho * dt)
            
            # Draw
            did_spike = np.random.random(N) < p
            
            if np.any(did_spike):
                n_s = np.sum(did_spike)
                A_rec[i] = n_s / (N * dt)
                ages[did_spike] = 0.0
                
        # PDE Simulation
        # Explicitly solve Age PDE with SAME hazard `rate`
        R_max = 65.0
        dr = 0.001
        pde = AgePDESolver(T, dt, R_max, dr, ref)
        
        rho_profile = np.zeros_like(pde.r_grid)
        mask = pde.r_grid > ref
        rho_profile[mask] = rate
        
        # Initial Condition matching MC?
        # MC starts with random ages. PDE starts with Gaussian?
        # Let's burn-in 10s.
        
        pde_rec = []
        for i in range(steps):
            pde.step(rho_profile)
            pde_rec.append(pde.get_mass() / dt) # Wait, get_mass?
            # get_diagnostics returns (mass, leak).
            # A(t) is returned by step()!
            # Oops `step` returns A_t.
            pass
            
        # Re-run PDE properly
        pde = AgePDESolver(T, dt, R_max, dr, ref)
        A_pde = []
        for i in range(steps):
             A_pde.append(pde.step(rho_profile))
             
        A_pde = np.array(A_pde)
        
        # Compare
        burn_steps = int(10.0/dt)
        A_mc_sm = ActivityMetrics.smooth_trace(A_rec[burn_steps:], dt, 0.02)
        A_pde_sm = ActivityMetrics.smooth_trace(A_pde[burn_steps:], dt, 0.02)
        
        L = min(len(A_mc_sm), len(A_pde_sm))
        diff = A_mc_sm[:L] - A_pde_sm[:L]
        
        rmse = np.sqrt(np.mean(diff**2))
        norm = np.mean(A_mc_sm[:L])
        nrmse = rmse / norm
        
        print(f"N={N}: NRMSE={nrmse:.4f}")
        stats.append({'N': N, 'nrmse': nrmse})
        
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(out_dir, 'renewal_benchmark.csv'), index=False)
    print("Saved benchmark.")

if __name__ == "__main__":
    benchmark_renewal()
