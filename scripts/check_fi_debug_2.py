import numpy as np
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from effective_neuron import EffectiveNeuron

def check():
    with open('configs/base.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    print(f"Params: th1={cfg['spike_gen']['theta_1']}, th2={cfg['spike_gen']['theta_2']}, kappa={cfg['neuron']['kappa']}")
    
    # Check +1 and -1
    for mu in [-1.0, 1.0]:
        ne = EffectiveNeuron(cfg, N=100)
        dt = 0.0001
        T = 2.0
        steps = int(T/dt)
        u = np.full(steps, mu)
        time = np.arange(steps)*dt
        
        v_sum = 0
        a_sum = 0
        spikes = 0
        
        # Burn in
        for i in range(1000):
            ne.step(0, dt, mu)
            
        for i in range(steps):
             s, _ = ne.step(time[i], dt, mu)
             v_sum += np.mean(ne.V)
             a_sum += np.mean(ne.a)
             spikes += np.sum(s)
             
        v_mean = v_sum / steps
        a_mean = a_sum / steps
        rate = spikes / (100 * T)
        
        exponent_est = cfg['spike_gen']['theta_0'] + cfg['spike_gen']['theta_1'] * v_mean + cfg['spike_gen']['theta_2'] * a_mean
        
        print(f"mu={mu}: Rate={rate:.2f}, V_mean={v_mean:.5f}, a_mean={a_mean:.5f}, ExpEst={exponent_est:.2f}")

if __name__ == "__main__":
    check()
