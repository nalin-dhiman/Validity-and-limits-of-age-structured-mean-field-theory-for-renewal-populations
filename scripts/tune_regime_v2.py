import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from effective_neuron import EffectiveNeuron
from utils import load_config

def run_sweep():
    with open('configs/base.yaml', 'r') as f:
        base_cfg = yaml.safe_load(f)
        
    dt = base_cfg['defaults']['dt']
    T = 20.0 # Fast check
    steps = int(T/dt)
    time = np.arange(steps)*dt
    N = 50 # Ensemble for smoothing
    
    # Try different theta_1 values
    theta_1_candidates = [500.0, 1000.0, 2000.0, 5000.0]
    results = []
    
    for th1 in theta_1_candidates:
        print(f"Testing theta_1 = {th1}")
        
        # 1. Tune theta_0 for Baseline ~ 5Hz
        # Rough binary search or sweep? Sweep is safer.
        # Estimate range: if th1=500, th0 might be ~ 1-5?
        # if th1=5000, th0 ~ 0.2.
        th0_sweep = np.linspace(-2, 5, 20)
        
        best_th0 = None
        best_err = 1e9
        
        for th0 in th0_sweep:
            cfg = base_cfg.copy()
            cfg['spike_gen']['theta_1'] = float(th1)
            cfg['spike_gen']['theta_0'] = float(th0)
            # Try to keep adaptation ratio? 
            # Previous: th2 = -24500 for th1=5000 (ratio ~ -5).
            # Let's keep ratio -5?
            cfg['spike_gen']['theta_2'] = float(-5.0 * th1)
            
            # Run Baseline (mu=0)
            ne = EffectiveNeuron(cfg, N=N)
            spk_count = 0
            # Sim
            u = np.zeros(steps)
            for i in range(steps):
                s, _ = ne.step(time[i], dt, u[i])
                spk_count += np.sum(s)
            
            rate = spk_count / (N * T)
            err = abs(rate - 7.5) # Target 7.5 Hz (center of 5-10)
            
            # print(f"  th0={th0:.2f}, Rate={rate:.2f}")
            
            if err < best_err:
                best_err = err
                best_th0 = th0
                
        print(f"  Best theta_0: {best_th0:.2f} (Rate error {best_err:.2f})")
        
        # 2. Check Slope (Rate at mu=1.0)
        cfg = base_cfg.copy()
        cfg['spike_gen']['theta_1'] = float(th1)
        cfg['spike_gen']['theta_0'] = float(best_th0)
        cfg['spike_gen']['theta_2'] = float(-5.0 * th1)
        
        ne = EffectiveNeuron(cfg, N=N)
        u = np.ones(steps) * 1.0 # mu=1.0
        spk_count = 0
        for i in range(steps):
            s, _ = ne.step(time[i], dt, u[i])
            spk_count += np.sum(s)
        rate_mu1 = spk_count / (N * T)
        
        print(f"  Rate(mu=1.0) = {rate_mu1:.2f} Hz")
        
        results.append({
            'theta_1': th1,
            'theta_0': best_th0,
            'theta_2': -5.0 * th1,
            'rate_0': 7.5, # approx
            'rate_1': rate_mu1
        })
        
    print("\nSummary:")
    print(pd.DataFrame(results))
    
    # Save best candidates
    pd.DataFrame(results).to_csv('results/tables/regime_candidates.csv', index=False)

if __name__ == "__main__":
    run_sweep()
