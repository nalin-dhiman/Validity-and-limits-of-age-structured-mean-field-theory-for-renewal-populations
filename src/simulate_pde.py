import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import load_config, set_seed, ensure_dir
from stimuli import StimulusGenerator
from age_pde import AgePDESolver
from age_adaptation_pde import AgeAdaptationPDESolver
from coupling import MeanFieldCoupling

def run_pde_simulation(config_path, output_file):
    cfg = load_config(config_path)
    set_seed(cfg['defaults']['seed'])
    
    dt = cfg['defaults']['dt']
    T = cfg.get('duration', 2.0)
    
    # Phase IX Check: R_max vs T
    # Minimum safe R_max is T + refractory + margin (e.g. 5s)
    margin = 5.0
    ref_period = cfg['spike_gen']['refractory']
    min_R = T + ref_period + margin
    
    R_max = cfg['pde']['R_max']
    # R_max Auto Logic (Simplified Match)
    if R_max == 'auto':
        # Just use a safe default for PDE-only, or estimate if possible.
        # Since we don't have MC pre-flight here easily without duplicating code,
        # we'll use conservative defaults for PDE-only.
        # User said "Add PDE-only... runs PDE reference... records mass/truncation telemetry"
        # We can try to use a generous R_max.
        R_max = 300.0 # Cap
        print(f"[simulate_pde] R_max set to auto: defaulted to {R_max} for PDE-only.")
        cfg['pde']['R_max'] = R_max 
    else:
        R_max = float(R_max)

    tol = 1e-6
    if R_max + tol < min_R:
        raise ValueError(f"CRITICAL: R_max ({R_max}) < Duration ({T}) + Ref ({ref_period}) + margin ({margin}) = {min_R}. This will cause mass leak. Increase R_max.")
    
    # Components
    # Phase IX: Age-Adaptation Closure
    pde = AgeAdaptationPDESolver(
        t_len=T, dt=dt, 
        R_max=cfg['pde']['R_max'], 
        dr=cfg['pde']['dr'], 
        params=cfg,
        init_mode='delta0' # Phase IX Strict Override
    )
    
    J = cfg['coupling']['J']
    coupler = MeanFieldCoupling(J, cfg['coupling']['tau_s'], dt)
    
    # Stimulus
    stim_gen = StimulusGenerator(dt)
    stim_params = cfg['stimulus'].copy()
    if 'type' in stim_params: del stim_params['type']
    u_ext = stim_gen.generate_ou(T, **stim_params)
    
    # Loop
    n_steps = len(u_ext)
    A_rec = np.zeros(n_steps)
    mass_rec = np.zeros(n_steps) # Diagnostic
    tail_rec = np.zeros(n_steps)
    
    # Metrics
    pde_metrics = {
        'mass_min': 1.0,
        'mass_final': 1.0,
        'tail_mass_max': 0.0,
        'cum_leak_final': 0.0,
        't_fail': -1.0,
        'R_max': R_max
    }
    
    for i in tqdm(range(n_steps), desc="PDE Solving"):
        # 1. Inputs
        u_total = u_ext[i] + coupler.output
        
        # 2. PDE Step (Dynamics + Hazard + Activity)
        A_t = pde.step(u_total)
        
        # Phase IX: Diagnostics (Dictionary return)
        diag = pde.get_diagnostics()
        mass = diag['mass']
        tail = diag['tail_mass']
        leak = diag['cum_leak']
        
        mass_rec[i] = mass
        tail_rec[i] = tail
        
        # Update Metrics
        pde_metrics['mass_min'] = min(pde_metrics['mass_min'], mass)
        pde_metrics['mass_final'] = mass
        pde_metrics['tail_mass_max'] = max(pde_metrics['tail_mass_max'], tail)
        pde_metrics['cum_leak_final'] = leak
        
        # Hard Guards (PDE-only should also fail if broken)
        if mass < 0.95:
             raise RuntimeError(f"PDE BREAKDOWN: Mass dropped to {mass:.4f} < 0.95 at t={i*dt:.3f}")
        if tail > 1e-3:
             raise RuntimeError(f"PDE BREAKDOWN: Tail mass rose to {tail:.4e} > 1e-3 at t={i*dt:.3f}")
        
        # 3. Coupling Update
        coupler.step(A_t)
        
        A_rec[i] = A_t
        
    np.savez(output_file, A=A_rec, u_ext=u_ext, mass=mass_rec, tail=tail_rec, config=cfg, pde_metrics=pde_metrics)

    print(f"Saved PDE results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--outfile', default='results/pde_sim.npz')
    args = parser.parse_args()
    
    run_pde_simulation(args.config, args.outfile)
