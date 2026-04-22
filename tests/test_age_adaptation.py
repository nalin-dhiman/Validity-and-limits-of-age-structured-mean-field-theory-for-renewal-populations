import sys
import os
import numpy as np
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from age_adaptation_pde import AgeAdaptationPDESolver

class TestAgeAdaptationPDE(unittest.TestCase):
    def test_mass_conservation(self):
        dt = 0.0001
        T = 1.0
        R_max = 10.0
        dr = 0.01
        
        params = {
            'neuron': {
                'tau_m': 0.02,
                'tau_a': 0.1,
                'kappa': 1.0,
                'sigma': 0.0
            },
            'spike_gen': {
                'refractory': 0.005,
                'theta_0': -20.0,
                'theta_1': 1.0,
                'theta_2': -1.0,
                'type': 'exponential'
            }
        }
        
        solver = AgeAdaptationPDESolver(T, dt, R_max, dr, params)
        
        # Run for a few steps with constant input
        u = 20.0 # High input to drive spiking
        
        masses = []
        for i in range(1000):
            A = solver.step(u)
            diag = solver.get_diagnostics()
            masses.append(diag['mass'])
            
        # Check conservation (Mass should be close to 1.0, minus leak)
        # Since R_max is smallish compared to infinite tail, leak might be non-zero.
        # But for 1s sim with R_max=10s, leak should be 0.
        
        self.assertTrue(abs(masses[-1] - 1.0) < 5e-4, f"Mass drifted: {masses[-1]}")
        
    def test_no_adaptation(self):
        # kappa=0, init m=0 -> m remains 0
        dt = 0.0001
        params = {
            'neuron': {'tau_m': 0.02, 'tau_a': 0.1, 'kappa': 0.0, 'sigma': 0.0},
            'spike_gen': {'refractory': 0.005, 'theta_0': -50.0, 'theta_1': 0.0, 'theta_2': 0.0, 'type': 'exponential'}
        }
        solver = AgeAdaptationPDESolver(1.0, dt, 10.0, 0.01, params)
        
        for i in range(100):
            solver.step(0.0)
            
        self.assertTrue(np.all(solver.m == 0.0), "m should remain 0 when kappa=0 and init=0")

if __name__ == '__main__':
    unittest.main()
