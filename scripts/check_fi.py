import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from effective_neuron import EffectiveNeuron

def check_fi():
    with open('configs/base.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    print(f"Checking F-I with: th0={cfg['spike_gen']['theta_0']}, th1={cfg['spike_gen']['theta_1']}, th2={cfg['spike_gen']['theta_2']}")
    
    mus = np.linspace(-2, 2, 21) # Narrow range for high gain
    rates = []
    
    dt = 0.0001
    T = 2.0
    steps = int(T/dt)
    time = np.arange(steps)*dt
    
    for mu in mus:
        ne = EffectiveNeuron(cfg, N=50) # Average 50 neurons
        u = np.full(steps, mu)
        spikes = 0
        for i in range(steps):
             s, _ = ne.step(time[i], dt, u[i])
             spikes += np.sum(s)
        r = spikes / (50 * T)
        rates.append(r)
        print(f"mu={mu:.2f}, Rate={r:.2f} Hz")
        
    plt.figure()
    plt.plot(mus, rates, 'o-')
    plt.xlabel('mu')
    plt.ylabel('Rate')
    plt.savefig('results/figures/fi_debug.png')

if __name__ == "__main__":
    check_fi()
