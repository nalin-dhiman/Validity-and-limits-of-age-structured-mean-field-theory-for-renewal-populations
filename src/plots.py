import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from metrics import ActivityMetrics

def load_data(path):
    return np.load(path, allow_pickle=True)

def plot_comparison_trace(mc_path, pde_path, out_path, title="Comparison"):
    mc = load_data(mc_path)
    pde = load_data(pde_path)
    
    A_mc = mc['A']
    A_pde = pde['A']
    dt = pde['config'].item()['defaults']['dt']
    
    # Check alignment
    if len(A_mc) != len(A_pde):
        # Assume MC is 1ms binned? No, sim_pop returns A aligned with input if binning logic was right.
        # But wait, simulate_pde returns array of size steps.
        # simulate_population returns binned?
        # Let's check sim_pop logic: returns A_rec binned at `bin_width=0.001`.
        # simulate_pde returns A_rec at speed dt=0.0001.
        # So PDE is 10x finer.
        
        mc_bin = 0.001
        pde_dt = dt
        
        # Downsample PDE to MC
        factor = int(mc_bin / pde_dt)
        # Reshape to (N_bins, Factor) and mean
        n_bins = len(A_mc)
        limit = n_bins * factor
        A_pde_ds = A_pde[:limit].reshape(n_bins, factor).mean(axis=1)
        time = np.arange(n_bins) * mc_bin
        
        # Smooth MC
        A_mc_smooth = ActivityMetrics.smooth_trace(A_mc, mc_bin, window_size=0.05)
        # Smooth PDE (Downsampled)
        A_pde_smooth = ActivityMetrics.smooth_trace(A_pde_ds, mc_bin, window_size=0.05)
    else:
        A_pde_ds = A_pde
        time = np.arange(len(A_mc)) * dt
        A_mc_smooth = ActivityMetrics.smooth_trace(A_mc, dt, window_size=0.05)
        A_pde_smooth = ActivityMetrics.smooth_trace(A_pde, dt, window_size=0.05)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, A_mc, label='MC Raw', alpha=0.3, color='gray', linewidth=0.5)
    plt.plot(time, A_mc_smooth, label='MC Smooth (50ms)', linewidth=2, color='C0')
    plt.plot(time, A_pde_smooth, label='PDE Smooth', linewidth=2, color='C1', linestyle='--')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Activity (Hz)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, time[-1])
    plt.savefig(out_path)
    plt.close()

def plot_single_trace(traces, out_path):
    """
    traces: list of dicts with keys: time, V, a, spikes, hazard, label
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for i, tr in enumerate(traces):
        ax = axes[i]
        t = tr['time']
        
        # Twin axis for Hazard?
        ax1 = ax
        ax2 = ax.twinx()
        
        # 1. Voltage
        ax1.plot(t, tr['V'], color='k', linewidth=0.8, label='V(t)')
        ax1.set_ylabel('Voltage (mV-like)', color='k')
        
        # 2. Adaptation and Hazard
        ax2.plot(t, tr['hazard'], color='r', alpha=0.6, linewidth=0.8, label='Hazard')
        ax2.set_ylabel('Hazard (Hz)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 3. Spikes
        for spsu in tr['spikes']:
            ax1.axvline(spsu, color='b', alpha=0.5, ymin=0.9, ymax=1.0)
            
        ax.set_title(tr['label'])
    
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_diagnostics(pde_path, out_path):
    pde = load_data(pde_path)
    mass = pde['mass']
    tail = pde.get('tail', np.zeros_like(mass))
    dt = pde['config'].item()['pde']['dr'] # No, dt is in defaults
    dt_sim = pde['config'].item()['defaults']['dt']
    time = np.arange(len(mass)) * dt_sim
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(time, mass, label='Mass')
    ax1.set_ylabel('Total Probability')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0.99, 1.01)
    
    ax2 = ax1.twinx()
    ax2.plot(time, tail, color='r', label='Tail Density q(R_max)')
    ax2.set_ylabel('Density at R_max', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('PDE Diagnostics: Mass Conservation & Leak')
    plt.savefig(out_path)
    plt.close()
    
def plot_failure_map2(result_grid, j_vals, c_vals, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(result_grid, xticklabels=np.round(c_vals, 2), yticklabels=np.round(j_vals, 2), cmap='viridis', annot=True, fmt='.2f')
    plt.xlabel('Shared Noise (c)')
    plt.ylabel('Coupling (J)')
    plt.title('RMSE (Smoothed MC vs PDE)')
    plt.savefig(out_path)
    plt.close()
