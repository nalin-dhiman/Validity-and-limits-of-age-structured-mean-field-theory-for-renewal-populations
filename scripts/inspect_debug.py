
import numpy as np
import os
import sys

def inspect():
    base = 'results/validate_scaling'

    # Import Metrics
    sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
    from metrics import ActivityMetrics
    
    dt = 0.0001
    tau = 0.05
    burn = 2000 # 0.2s
    
    # 1. PDE
    pde_file = os.path.join(base, 'pde_ref_age_adapt_jensen.npz')
    if not os.path.exists(pde_file):
        print(f"No PDE file: {pde_file}")
        return
        
    pde_raw = np.load(pde_file)['A']
    # Smooth
    pde = ActivityMetrics.smooth_trace(pde_raw, dt, tau)
    pde = pde[burn:] 
    
    print(f"PDE Mean: {np.mean(pde):.4f}")
    print(f"PDE Std:  {np.std(pde):.4f}")
    
    # 2. MC
    mc_files = [f for f in os.listdir(base) if f.startswith('mc_') and f.endswith('.npz')]
    
    for f in sorted(mc_files):
        path = os.path.join(base, f)
        mc_raw = np.load(path)['A']
        mc = ActivityMetrics.smooth_trace(mc_raw, dt, tau)
        mc = mc[burn:]
        
        # Align
        L = min(len(pde), len(mc))
        diff = mc[:L] - pde[:L]
        
        bias = np.mean(diff)
        rmse = np.sqrt(np.mean(diff**2))
        crmse = np.sqrt(np.var(diff)) # Centered RMSE (Std of residual)
        
        print(f"--- {f} ---")
        print(f"MC Mean: {np.mean(mc):.4f}")
        print(f"MC Std:  {np.std(mc):.4f}")
        print(f"Bias:    {bias:.4f}")
        print(f"RMSE:    {rmse:.4f}")
        print(f"CRMSE:   {crmse:.4f}")
        print(f"NormRMSE_std: {rmse/np.std(pde):.4f}")

if __name__ == "__main__":
    inspect()
