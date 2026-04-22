import sys
import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import load_config, set_seed
from effective_neuron import EffectiveNeuron # We'll need to modify or use vectorization
from stimuli import StimulusGenerator
from metrics import ActivityMetrics
from coupling import MeanFieldCoupling
from hazards import HazardFunction
from age_pde import AgePDESolver
from age_adaptation_pde import AgeAdaptationPDESolver
from pde_errors import PDESafetyError
from scipy.ndimage import gaussian_filter1d
from numerics_numba import step_neuron_numba
from simulation_logger import SimulationLogger
import traceback

# Optimization: Define a VectorizedNeuron class inside here for speed
class VectorizedNeuron:
    def __init__(self, params, N, seed=None):
        self.params = params
        self.N = N
        p = params['neuron']
        self.tau_m = p['tau_m']
        self.tau_a = p['tau_a']
        self.kappa = p['kappa']
        self.sigma = p['sigma']
        
        sg = params['spike_gen']
        self.refractory = sg['refractory']
        self.hazard_fn = HazardFunction(params)
        
        self.V = np.zeros(N)
        self.a = np.zeros(N)
        self.last_spike_time = np.full(N, -1000.0)
        
        if seed is not None:
            np.random.seed(seed)
            
        self.theta_0 = sg['theta_0']
        self.theta_1 = sg['theta_1']
        self.theta_2 = sg['theta_2']
        type_str = sg.get('type', 'exponential')
        self.h_type_code = 1 if type_str == 'softplus' else 0
            
    def step(self, t, dt, u_t, shared_noise=0.0, c_sqrt_shared=0.0, c_sqrt_indep=1.0):
        # Numba Optimized Step
        # noise is generated inside the kernel to avoid allocating N-sized arrays in Python
        
        spikes, lam = step_neuron_numba(
            self.V, self.a, self.last_spike_time, t, dt, u_t,
            shared_noise, self.N,
            self.tau_m, self.tau_a, self.kappa, self.sigma,
            self.refractory,
            self.theta_0, self.theta_1, self.theta_2, self.h_type_code,
            c_sqrt_shared, c_sqrt_indep
        )
            
        return spikes, lam

def run_population_sim(config, output_file, online_pde=False):
    # Initialize Logger
    logger = SimulationLogger('simulate_population.py', config, output_file)
    
    try:
        _run_population_sim_inner(config, output_file, online_pde)
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        traceback.print_exc()
        logger.close(exit_code=1)
        sys.exit(1)
    else:
        logger.close(exit_code=0)

def _run_population_sim_inner(config, output_file, online_pde=False):
    # Setup
    dt = config['defaults']['dt']
    T = 2.0 # Default short run
    if 'duration' in config: T = config['duration']
    
    N = config['population']['N']
    N = config['population']['N']
    # if N_override: N = N_override (Removed)
    
    c = config['population']['shared_noise_fraction']
    J = config['coupling']['J']
    
    # Components
    stim_gen = StimulusGenerator(dt)
    stim_params = config['stimulus'].copy()
    if 'type' in stim_params: del stim_params['type']
    u_ext = stim_gen.generate_ou(T, **stim_params) # External drive
    
    pop = VectorizedNeuron(config, N, seed=config['defaults']['seed'])
    coupler = MeanFieldCoupling(J, config['coupling']['tau_s'], dt)
    
    # Pre-generate noise for efficiency
    n_steps = len(u_ext)
    
    # Noise mixing
    # sqrt(c) * shared + sqrt(1-c) * indep
    # We generate these on the fly or pre-gen? 
    # N = config['population']['N'] (Already set above)
    
    # Check c
    
    pop = VectorizedNeuron(config, N, seed=config['defaults']['seed'])
    J = config['coupling']['J']
    tau_s = config['coupling']['tau_s']
    coupler = MeanFieldCoupling(J, tau_s, dt)
    
    c = config['population'].get('shared_noise_fraction', 0.0)
    
    # Initialize Population State validly (ages < R_max)
    # Uniform ages [0, 0.010] (sync start with spread) - MATCHES PDE delta0
    # Updates pop.last_spike_time
    # Initialize all spikes to be very recent (0 to 10ms ago)
    pop.last_spike_time = -np.random.uniform(0, 0.010, N)
    
    # Online PDE Setup
    pde_age = None
    pde_adapt = None
    pde_age_rec = []
    pde_adapt_rec = []
    
    save_fields = config.get('save_fields', False)
    rho_snap = [] if (online_pde and save_fields) else None
    
    if online_pde:
        R_max = config['pde']['R_max']
        dr = config['pde']['dr']
        ref_period = config['spike_gen']['refractory']
        
        # Phase IX Check: R_max vs T
        # Minimum safe R_max is T + refractory + margin (e.g. 5s)
        margin = 5.0
        
        # Track Active Radius for Efficiency
        # Initially just refractory + small margin
        sim_active_r = 0.05
        
        if R_max == 'auto':
            # Auto-calculate safe R_max based on estimated firing rate
            # Phase IX Hardening: Use pre-flight estimate
            est_rate = estimate_firing_rate(config)
            mean_isi = 1.0 / max(est_rate, 1e-4) # Safety floor
            
            # Criteria:
            # Criteria (Phase IX Hardening):
            # 1. T + ref + 15 (Base safety margin)
            # 2. 10 * mean_isi (Statistical support)
            # 3. Floor of T + ref + 30 (User specified constraint ?? - actually let's assume max(T+ref+15, 10*isi) bounded by 300, and maybe floor 30.
            # User instruction: R_max = max(T + ref + 15, 10*mean_isi, T + ref + 30) -> equivalent to max(10*mean_isi, T+ref+30).
            # Also cap at 300.
            
            val_isi = 10.0 * mean_isi
            val_base = T + ref_period + 30.0
            
            R_calc = max(val_base, val_isi)
            R_max = min(R_calc, 300.0) # Cap
            
            print(f"[simulate_population] R_max set to auto: calculated {R_max:.4f} (T={T}, EstRate={est_rate:.2f}Hz, ISI={mean_isi:.2f}s)")
        else:
             R_max = float(R_max) # Ensure float works if yaml parsed as string

        # We need the resolved grid for age indexing.
        M = int(R_max / dr)
        r_grid_full = np.linspace(0, R_max, M + 1)

        # Consistency Check
        min_R = T + ref_period + margin
        
        # Guard with tolerance (Step 2)
        tol = 1e-6
        if R_max + tol < min_R:
             # Just warning now if "auto" might have picked something else, but we want to fail if explicit
             # If explicit R_max is too small, fail.
             if config['pde']['R_max'] != 'auto':
                raise ValueError(f"CRITICAL: R_max ({R_max}) < Duration ({T}) + Ref ({ref_period}) + margin ({margin}) = {min_R}. This will cause mass leak.\nFix: Set pde.R_max >= {min_R + 1.0:.3f} or use 'auto' in config.")
        
        # 1. Age-Only (Diagnostic)
        pde_age = AgePDESolver(T, dt, R_max, dr, ref_period)
        
        # 2. Age-Adaptation (Generative)
        use_jensen = config['pde'].get('use_jensen', True)
        pde_adapt = AgeAdaptationPDESolver(T, dt, R_max, dr, config, use_jensen=use_jensen)
        
        # Grid from adapt (they should match)
        r_grid = pde_adapt.r_grid
        n_bins_pde = len(r_grid)
    
    # Records
    A_rec = np.zeros(n_steps)
    
    # Phase IX: Diagnostics Tracking
    pde_metrics = {
        'mass_min': 1.0,
        'mass_final': 1.0,
        'tail_mass_max': 0.0,
        'cum_leak_final': 0.0,
        't_fail': -1.0,
        'R_max': R_max if online_pde else 0.0
    }
    
    for i in tqdm(range(n_steps), desc=f"Simulating N={N}"):
        t = i * dt
        
        # 1. Coupling
        u_total = u_ext[i] + coupler.output
        
        # 2. Noise Parameters
        xi_shared = np.random.normal(0, 1)
        c_sqrt_shared = np.sqrt(c)
        c_sqrt_indep = np.sqrt(1-c)
        
        # 3. Step
        # Hazard Inference Part 1 (for Age-Only Diagnostic)
        if online_pde:
            ages = t - pop.last_spike_time
        # Hazard Inference Part 1 (for Age-Only Diagnostic)
        if online_pde:
            ages = t - pop.last_spike_time
            bin_idx = (ages / dr).astype(int)
            # Optimization: bin_idx limited by active_r roughly, but we need full n_risk for density? 
            # Actually we only need n_risk where we compute rho.
            # And we only compute rho up to limit_idx.
            # So we can clip bin_idx? 
            # If neuron has age > limit_idx, it contributes to risk at that age, but we don't care about rho there yet?
            # Actually if we ignore them, n_risk counts will be correct at lower bins.
            # We just need to ensure bincount is efficient.
            # bincount on array of size N is fine. minlength matters.
            # We can use minlength=limit_idx? 
            # But if bin_idx > limit_idx, bincount will extend if we don't clip?
            # Or we filter `bin_idx < limit_idx`.
            
            limit_idx_risk = int(sim_active_r / dr) + 20
            # Ensure limit_idx_risk matches what we use in Part 2. same logic.
            # Let's just use a safe upper bound for now or recalc in Part 2.
            # Actually Part 1 just computes n_risk. 
            
            # Perform full bincount but limit minlength?
            # If we limit minlength, and max(bin_idx) > minlength, result is larger.
            # We want to slice n_risk later. So we need at least limit_idx.
            # Let's use full bincount (N=5000 is small), but then slice n_risk.
            # Wait, allocating bincount(300k) is the issue? 300k ints is 1.2MB. Fast.
            # The issue was gaussian_filter on 300k.
            
            bin_idx = np.clip(bin_idx, 0, n_bins_pde-1)
            n_risk = np.bincount(bin_idx, minlength=n_bins_pde)
            
            # NOTE: We optimize Part 2 (Hazard Calc), but keep Part 1 (Risk counting) simple as bincount is fast.
            # Optimizing bincount size:
            # If we know max age is t, we can use dynamic size.
            # sim_active_r tracks t.
            # So max(bin_idx) is around sim_active_r/dr.
            # So bincount size naturally grows with t.
            # But we used minlength=n_bins_pde (fixed 300k).
            # We can change minlength?
            # Let's leave n_risk size large (allocation is cheap once/cached? No numpy allocates new).
            # bincount return size depends on max(bin_idx) and minlength.
            # If we set minlength=0, size is max(bin_idx)+1.
            # We should let it grow naturally!
            # bin_idx = (ages / dr).astype(int) -> max is t/dr.
            # n_risk = np.bincount(bin_idx).
            # Then pad/slice in Part 2.
            
            n_risk = np.bincount(bin_idx) # Let it be natural size
            # We need to ensure we have enough bins for limit_idx in Part 2.
            # We handle that in Part 2.
            
        spikes, _ = pop.step(t, dt, u_total, shared_noise=xi_shared, 
                             c_sqrt_shared=c_sqrt_shared, c_sqrt_indep=c_sqrt_indep)
        
        if online_pde:
            # Hazard Inference Part 2
            
            # Optimization: Slice based on sim_active_r
            sim_active_r = min(R_max, sim_active_r + dt * 1.2)
            limit_idx = int(sim_active_r / dr) + 20
            limit_idx = min(limit_idx, n_bins_pde)
            
            # Slice operations
            
            spiking_ages = ages[spikes]
            if len(spiking_ages) > 0:
                spk_idx = (spiking_ages / dr).astype(int)
                # Filter out of bounds (shouldn't happen with correct active_r but safety)
                mask_in = spk_idx < limit_idx
                spk_idx = spk_idx[mask_in]
                n_spk = np.bincount(spk_idx, minlength=limit_idx)
            else:
                n_spk = np.zeros(limit_idx)
            
            # Slice n_risk using pad/slice logic
            if len(n_risk) < limit_idx:
                n_risk_slice = np.pad(n_risk, (0, limit_idx - len(n_risk)), 'constant')
            else:
                n_risk_slice = n_risk[:limit_idx]
            
            # Naive Hazard (Sliced)
            rho_est = np.divide(n_spk, n_risk_slice * dt + 1e-12)
            
            # Smooth (Sliced)
            rho_smooth = gaussian_filter1d(rho_est, sigma=2.0)
            rho_smooth = np.clip(rho_smooth, 0.0, 200.0)
            
            # Refractory on slice
            r_grid_slice = r_grid_full[:limit_idx]
            rho_smooth[r_grid_slice < ref_period] = 0.0
            
            # Step Age-Only (Diagnostic)
            # Pass sliced rho_smooth. AgePDESolver handles it.
            a_age = pde_age.step(rho_smooth)
            pde_age_rec.append(a_age)
            
            # Step Age-Adaptation (Generative)
            a_adapt = pde_adapt.step(u_total)
            pde_adapt_rec.append(a_adapt)
            
            # Snapshots (Adapt Model)
            if i % 100 == 0 and save_fields:
                 if rho_snap is None: rho_snap = []
                 # Compute current hazard profile from model state
                 v_int = pde_adapt.v
                 m_int = pde_adapt.m
                 rho_model = pde_adapt.compute_hazard(v_int, m_int)
                 rho_snap.append(rho_model.copy())
            
            # Phase IX: Periodic Conservation Check (e.g., every 100 steps)
            if i % 100 == 0:
                # Get Diagnostics (Dicts)
                d1 = pde_age.get_diagnostics() 
                d2 = pde_adapt.get_diagnostics()
                
                # Unpack
                m1, m2 = d1['mass'], d2['mass']
                t1, t2 = d1['tail_mass'], d2['tail_mass']
                l1, l2 = d1['cum_leak'], d2['cum_leak']
                
                # Track Min/Max
                curr_min_mass = min(m1, m2)
                curr_max_tail = max(t1, t2)
                curr_leak = max(l1, l2) # Conservatively take max leak
                
                pde_metrics['mass_min'] = min(pde_metrics['mass_min'], curr_min_mass)
                pde_metrics['mass_final'] = curr_min_mass
                pde_metrics['tail_mass_max'] = max(pde_metrics['tail_mass_max'], curr_max_tail)
                pde_metrics['cum_leak_final'] = curr_leak
                
                # Hard Guards
                # Mass < 0.95 -> Breakdown
                if curr_min_mass < 0.95:
                    pde_metrics['t_fail'] = t
                    raise RuntimeError(f"PDE BREAKDOWN: Mass dropped to {curr_min_mass:.4f} < 0.95 at t={t:.3f}")
                
                # Tail Mass > 1e-3 (Hard) -> Breakdown    
                if curr_max_tail > 1e-3:
                    pde_metrics['t_fail'] = t
                    raise RuntimeError(f"PDE BREAKDOWN: Tail mass rose to {curr_max_tail:.4e} > 1e-3 at t={t:.3f}")
                    
                # Soft warnings are just recorded in metrics and handled by classifier
        
        # spikes was already computed at line 168. DO NOT CALL step() AGAIN.
        
        # 4. Metrics
        n_curr = np.sum(spikes)
        A_inst = n_curr / (N * dt)
        coupler.step(A_inst)
        A_rec[i] = A_inst
        
    # Save
    save_dict = {
        'A': A_rec, 'u_ext': u_ext, 'config': config,
        'pop_V': pop.V, 'pop_last_spike_time': pop.last_spike_time,
        'pde_metrics': pde_metrics 
    }
    if online_pde:
        save_dict['A_pde_age_only'] = np.array(pde_age_rec)
        save_dict['A_pde_adapt'] = np.array(pde_adapt_rec)
        # Compatibility alias
        save_dict['A_pde'] = np.array(pde_adapt_rec)
        
        if rho_snap is not None:
             save_dict['rho_field'] = np.array(rho_snap) # [Time/100, R]
             save_dict['r_grid'] = r_grid
             
    np.savez(output_file, **save_dict)
    print(f"Saved results to {output_file}")


def estimate_firing_rate(config):
    """
    Run a short 1s simulation to estimate firing rate/ISI for R_max scaling.
    Returns: median_rate (Hz)
    """
    print("[Pre-Flight] Estimating firing rate...")
    dt = config['defaults']['dt']
    T_burn = 1.0
    N = 1000 # Enough for estimate
    
    # Minimal components
    stim_gen = StimulusGenerator(dt)
    stim_params = config['stimulus'].copy()
    if 'type' in stim_params: del stim_params['type']
    u_ext = stim_gen.generate_ou(T_burn, **stim_params)
    
    # Temporary Neuron Group
    pop = VectorizedNeuron(config, N, seed=12345)
    
    J = config['coupling']['J']
    tau_s = config['coupling']['tau_s']
    coupler = MeanFieldCoupling(J, tau_s, dt)
    
    c = config['population'].get('shared_noise_fraction', 0.0)
    
    # Initialize
    pop.last_spike_time = -np.random.uniform(0, 0.010, N)
    
    n_steps = len(u_ext)
    A_trace = []
    
    # Simple Loop
    for i in range(n_steps):
        t = i * dt
        u_total = u_ext[i] + coupler.output
        
        xi_shared = np.random.normal(0, 1)
        c_sqrt_shared = np.sqrt(c)
        c_sqrt_indep = np.sqrt(1-c)
        
        spikes, _ = pop.step(t, dt, u_total, shared_noise=xi_shared, 
                             c_sqrt_shared=c_sqrt_shared, c_sqrt_indep=c_sqrt_indep)
        
        n_curr = np.sum(spikes)
        A_inst = n_curr / (N * dt)
        coupler.step(A_inst)
        
        if i > n_steps // 4: # Ignore first 250ms transient
            A_trace.append(A_inst)
            
    # Rate Estimate Robustly
    if len(A_trace) > 0:
        rate_est = np.median(A_trace)
    else:
        rate_est = 10.0 # Fallback
    
    print(f"[Pre-Flight] Estimated Median Rate: {rate_est:.2f} Hz")
    return rate_est



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--outfile', default='results/pop_sim.npz')
    parser.add_argument('--online-pde', action='store_true')
    parser.add_argument('--save-fields', action='store_true', help="Save full PDE fields (warning: large files)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    cfg['save_fields'] = args.save_fields
    run_population_sim(cfg, args.outfile, args.online_pde)
