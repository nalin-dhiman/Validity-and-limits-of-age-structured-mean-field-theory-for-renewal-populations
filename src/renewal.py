import numpy as np
from hazards import HazardFunction

class RenewalModel:
    def __init__(self, params):
        self.params = params
        p = params['neuron']
        self.tau_m = p['tau_m']
        self.tau_a = p['tau_a']
        self.kappa = p['kappa']
        self.sigma = p['sigma'] # Base sigma
        
        self.hazard_fn = HazardFunction(params)
        
        # State (Mean V, Mean a)
        self.V = 0.0
        self.a = 0.0
        
    def reset(self):
        self.V = 0.0
        self.a = 0.0
        
    def step(self, dt, u_total, shared_noise_term=0.0, shared_fraction=0.0):
        """
        Update mean state and return instantaneous hazard potential.
        shared_noise_term: Standard normal sample (scalar)
        shared_fraction: c parameter
        
        We simulate the "Representative Neuron" driven by u and shared noise.
        """
        
        # Effective noise on the MEAN variable
        eff_sigma = self.sigma * np.sqrt(shared_fraction)
        
        sqrt_dt = np.sqrt(dt)
        dV = (-self.V / self.tau_m - self.a + u_total) * dt + eff_sigma * sqrt_dt * shared_noise_term
        da = (-self.a / self.tau_a + self.kappa * self.V) * dt
        
        self.V += dV
        self.a += da
        
        lam = self.hazard_fn(self.V, self.a)
        return lam
