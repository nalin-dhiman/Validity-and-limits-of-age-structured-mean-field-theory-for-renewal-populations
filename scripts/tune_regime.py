import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from effective_neuron import EffectiveNeuron

def tune_regime():
    # Load base config
    with open('configs/base.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Option A: High Gain
    cfg['spike_gen']['theta_1'] = 5000.0 # Increase gain
    cfg['spike_gen']['theta_2'] = -24500.0 # Match adaptation (0.2 ratio)
    
    # Sweep theta_0 to find rate ~ 5Hz
    # With strong adaptation, theta_0 might need to be higher to overcome it?
    # Or adaptation is zero at baseline if u=0?
    # V fluctuates. "a" fluctuates.
    # We'll sweep wide.
    theta_0_vals = np.linspace(-50, 50, 21)
    rates = []
    
    print("Sweeping theta_0 for baseline (mu=0) rate ~ 5Hz...")
    
    for th0 in theta_0_vals:
        cfg['spike_gen']['theta_0'] = float(th0)
        
        # Run short sim
        ne = EffectiveNeuron(cfg, N=100) # 100 neurons for mean rate
        # 10s simulation
        dt = 0.0001
        T = 5.0
        steps = int(T/dt)
        # mu = 0
        u = np.zeros(steps)
        time = np.arange(steps)*dt
        
        spikes = 0
        for i in range(steps):
             s, _ = ne.step(time[i], dt, u[i])
             spikes += np.sum(s)
             
        r = spikes / (100 * T)
        rates.append(r)
        print(f"theta_0={th0:.2f}, Rate={r:.2f} Hz")
        
    # Plot
    plt.figure()
    plt.plot(theta_0_vals, rates, 'o-')
    plt.axhline(5.0, color='r', linestyle='--')
    plt.xlabel('theta_0')
    plt.ylabel('Rate (Hz)')
    plt.title('Regime Tuning')
    plt.grid(True)
    plt.savefig('results/figures/tuning_curve.png')
    
    # Pick best theta_0
    # Interpolate
    target = 5.0
    # simple closest
    idx = np.argmin(np.abs(np.array(rates) - target))
    best_th0 = theta_0_vals[idx]
    print(f"Best theta_0 approx: {best_th0}")
    
    # Now check F-I range with this theta_0
    cfg['spike_gen']['theta_0'] = float(best_th0)
    
    mu_sweep = np.linspace(-5, 10, 10)
    fi_rates = []
    print("Checking F-I Range...")
    for mu in mu_sweep:
        ne = EffectiveNeuron(cfg, N=50) 
        u = np.full(steps, mu)
        spikes = 0
        for i in range(steps):
             s, _ = ne.step(time[i], dt, u[i])
             spikes += np.sum(s)
        r = spikes / (50 * T)
        fi_rates.append(r)
        
    plt.figure()
    plt.plot(mu_sweep, fi_rates, 'x-')
    plt.xlabel('Input Mean')
    plt.ylabel('Rate (Hz)')
    plt.title(f'F-I Curve (th0={best_th0:.2f}, th1={cfg["spike_gen"]["theta_1"]})')
    plt.grid(True)
    plt.savefig('results/figures/tuning_fi.png')
    print(f"F-I Range: {min(fi_rates):.2f} - {max(fi_rates):.2f} Hz")

if __name__ == "__main__":
    tune_regime()
