import numpy as np
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from effective_neuron import EffectiveNeuron

def debug():
    with open('configs/base.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['spike_gen']['theta_1'] = 5000.0
    cfg['spike_gen']['theta_0'] = -100.0
    
    ne = EffectiveNeuron(cfg, N=1)
    
    dt = 0.0001
    u = 0.0
    
    print("Debugging Neuron Step...")
    for i in range(100):
        # manually run step parts or just run step and spy
        # Let's peek at V before step
        v_old = ne.V[0]
        spk, lam = ne.step(i*dt, dt, u)
        
        # Re-calc exponent to verify
        # exp = th0 + th1*V + th2*a
        exponent = cfg['spike_gen']['theta_0'] + cfg['spike_gen']['theta_1'] * ne.V[0] + cfg['spike_gen']['theta_2'] * ne.a[0]
        
        if i % 10 == 0:
            print(f"t={i*dt:.4f} V={ne.V[0]:.4f} a={ne.a[0]:.4f} Exp={exponent:.2f} Lam={lam[0]:.2f}")
            
if __name__ == "__main__":
    debug()
