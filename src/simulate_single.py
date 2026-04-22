import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils import load_config, set_seed, ensure_dir
from effective_neuron import EffectiveNeuron
from stimuli import StimulusGenerator
from plots import plot_single_trace

def run_simulation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--out_dir', type=str, default='results/figures')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    ensure_dir(args.out_dir)
    set_seed(cfg['defaults']['seed'])
    
    dt = cfg['defaults']['dt']
    T = 2.0 # 2 seconds for traces
    

    
    # 1. Generate Traces (Fig A1)
    stim_gen = StimulusGenerator(dt)
    
    # Create stimuli
    stim_params = cfg['stimulus'].copy()
    if 'type' in stim_params: del stim_params['type']
    
    u_ou = stim_gen.generate_ou(T, **stim_params)
    u_band = stim_gen.generate_band_limited(T, f_c=50.0, sigma_u=cfg['stimulus']['sigma_u'])
    
    params_pre = cfg['stimulus'].copy()
    if 'type' in params_pre: del params_pre['type']
    params_post = cfg['stimulus'].copy()
    if 'type' in params_post: del params_post['type']
    params_post['sigma_u'] *= 2.0 
    u_nonstat = stim_gen.generate_non_stationary(T, params_pre, params_post)
    
    stimuli = {'OU': u_ou, 'Band-Limited': u_band, 'Non-Stationary': u_nonstat}
    trace_data = []
    
    for name, u in stimuli.items():
        neuron = EffectiveNeuron(cfg, N=1)
        time = np.arange(len(u)) * dt
        
        V_rec = []
        a_rec = []
        haz_rec = []
        spikes_rec = []
        
        for idx, t in enumerate(time):
            spk, lam = neuron.step(t, dt, u[idx])
            V_rec.append(neuron.V[0])
            a_rec.append(neuron.a[0])
            haz_rec.append(lam[0])
            if spk[0]:
                spikes_rec.append(t)
                
        trace_data.append({
            'time': time, 'V': V_rec, 'a': a_rec, 
            'hazard': haz_rec, 'spikes': spikes_rec, 'label': name
        })
            
    plot_single_trace(trace_data, os.path.join(args.out_dir, 'Fig_A1_Traces.png'))
    print("Generated Fig A1")
    
    # 2. F-I Curve (Fig A2)
    # Detailed sweep for Continuous Regime analysis
    # Range: [-5..-1] step 1, [-1..2] step 0.1, [2..10] step 1
    mus_detailed = np.concatenate([
        np.arange(-5.0, -1.0, 1.0),
        np.arange(-1.0, 2.05, 0.1),
        np.arange(3.0, 11.0, 1.0)
    ])
    mus_detailed = np.unique(mus_detailed)
    n_mus = len(mus_detailed)
    
    T_fi = 60.0 # Requirement: 60s
    n_seeds = 5 # Requirement: >= 5 seeds
    
    total_neurons = n_mus * n_seeds
    
    print(f"Computing F-I Curve ({n_mus} points, {n_seeds} seeds) - Vectorized N={total_neurons}...")
    
    # Create Neuron Group
    neuron_pop = EffectiveNeuron(cfg, N=total_neurons)
    
    # Prepare Inputs
    # We need u_ext of shape (total_neurons, steps)
    # Mapping: neuron i -> (mu_idx, seed_idx)
    # i = mu_idx * n_seeds + seed_idx
    
    steps = int((T_fi + 10.0) / dt) # 10s burn-in
    time = np.arange(steps) * dt
    
    u_full = np.zeros((total_neurons, steps))
    
    mu_map = [] # stores mu value for each neuron
    
    for m_i, mu in enumerate(mus_detailed):
        for s in range(n_seeds):
            idx = m_i * n_seeds + s
            u_full[idx, :] = mu
            mu_map.append(mu)
            
    mu_map = np.array(mu_map)
    
    # Run Simulation
    spike_counts = np.zeros(total_neurons)
    
    # Manual Loop
    burn_steps = int(10.0 / dt)
    
    # Batch process? No, step is time-sequential.
    # But vectorized over N.
    
    for i in tqdm(range(steps)):
        spk, _ = neuron_pop.step(time[i], dt, u_full[:, i])
        if i >= burn_steps:
             spike_counts += spk.astype(float)
             
    rates = spike_counts / T_fi
    
    # Collect Results
    results = []
    for i in range(total_neurons):
        results.append({
            'mu': mu_map[i],
            'seed': i % n_seeds, # seed index, actual seed logic handled by EffectiveNeuron?
            # EffectiveNeuron init with 1 seed? 
            # It generates vector xi. So randomness is independent across N. Good.
            'rate': rates[i]
        })
        
    # Save detailed
    df_fi = pd.DataFrame(results)
    df_fi.to_csv('results/tables/fi_curve_seeds.csv', index=False)
    
    # Summary
    df_summ = df_fi.groupby('mu')['rate'].agg(['mean', 'std']).reset_index()
    df_summ.rename(columns={'mean': 'mean_rate', 'std': 'std_rate'}, inplace=True)
    df_summ.to_csv('results/tables/fi_curve_summary.csv', index=False)
    
    # Print sample
    print("F-I Curve Sample points:")
    print(df_summ[df_summ['mu'].isin([0.0, 1.0])])
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(df_summ['mu'], df_summ['mean_rate'], yerr=df_summ['std_rate'], fmt='o-', capsize=3, label='Mean rate')
    plt.xlabel('Input Mean $\mu_u$')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('F-I Curve (Continuous Regime)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'Fig_A2_FI_Curve.png'))
    plt.close()
    
    print("Generated Fig A2")

    # 3. ISI Histogram (Fig A3)
    T_isi = 100.0 # Longer for stats
    u_isi = stim_gen.generate_ou(T_isi, **stim_params)
    time = np.arange(len(u_isi)) * dt
    neuron = EffectiveNeuron(cfg, N=1)
    spikes = []
    
    for idx, t in enumerate(time):
        spk, _ = neuron.step(t, dt, u_isi[idx])
        if spk[0]: spikes.append(t)
        
    isis = np.diff(spikes)
    
    plt.figure()
    sns.histplot(isis, bins=50, kde=False)
    plt.xlabel('ISI (s)')
    plt.title(f'ISI Histogram (Mean ISI={np.mean(isis):.3f}s)')
    plt.savefig(os.path.join(args.out_dir, 'Fig_A3_ISI.png'))
    plt.close()
    
    # Save Stats
    if len(isis) > 0:
        pd.DataFrame({'isi': isis}).to_csv('results/tables/isi_stats.csv', index=False)
    print("Generated Fig A3")

if __name__ == "__main__":
    run_simulation()
