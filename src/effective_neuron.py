import numpy as np
from hazards import HazardFunction

# Numba Optimization
try:
    from numba_kernels import step_neuron_numba
    use_numba = True
except ImportError:
    use_numba = False

class EffectiveNeuron:
    def __init__(self, params, N=1, seed=None):
        self.params = params
        self.N = N
        
        # Parameters
        self.tau_m = params['neuron']['tau_m']
        self.tau_a = params['neuron']['tau_a']
        self.kappa = params['neuron']['kappa']
        self.sigma = params['neuron']['sigma']
        
        self.refractory = params['spike_gen']['refractory']
        self.hazard_fn = HazardFunction(params)
        
        # State
        self.V = np.zeros(N)
        self.a = np.zeros(N)
        self.last_spike_time = np.full(N, -1000.0) # Long ago
        
        if seed is not None:
            np.random.seed(seed)
            
    def reset(self):
        self.V = np.zeros(self.N)
        self.a = np.zeros(self.N)
        self.last_spike_time = np.full(self.N, -1000.0)

    def step(self, t, dt, u_t, noise_term=None):
        """
        Full step: Dynamics -> Hazard -> Spikes
        u_t: (N,) or scalar
        noise_term: (N,) pre-generated noise or None to generate internally
        """
        
        # 1. Dynamics (Euler-Maruyama)
        # V_{t+dt} = V + (-V/tau_m - a + u)*dt + sigma*sqrt(dt)*xi
        # a_{t+dt} = a + (-a/tau_a + kappa*V)*dt
        
        if noise_term is None:
            xi = np.random.normal(0, 1, self.N)
        else:
            xi = noise_term
            
        # 1. Dynamics (Euler-Maruyama) & 2. Spike Generation
        # V_{t+dt} = V + (-V/tau_m - a + u)*dt + sigma*sqrt(dt)*xi
        # a_{t+dt} = a + (-a/tau_a + kappa*V)*dt
        
        if noise_term is None:
            xi = np.random.normal(0, 1, self.N)
        else:
            xi = noise_term
            
        if use_numba:
             # Ensure u_t is array
             if np.isscalar(u_t):
                 u_val = np.full(self.N, u_t)
             else:
                 u_val = u_t
                 
             spikes, lam = step_neuron_numba(
                 self.V, self.a, self.last_spike_time, dt, t, u_val,
                 self.sigma, self.kappa, self.tau_m, self.tau_a, self.refractory,
                 self.params['spike_gen']['theta_0'],
                 self.params['spike_gen']['theta_1'],
                 self.params['spike_gen']['theta_2'],
                 xi
             )
             return spikes, lam
             
        # Fallback Python
        sqrt_dt = np.sqrt(dt)
        
        dV = (-self.V / self.tau_m - self.a + u_t) * dt + self.sigma * sqrt_dt * xi
        da = (-self.a / self.tau_a + self.kappa * self.V) * dt
        
        self.V += dV
        self.a += da
        
        # 2. Spike Generation
        # lambda = hazard(V, a)
        # Refractory check
        
        is_ref = (t - self.last_spike_time) < self.refractory
        
        # Calculate hazard
        lam = self.hazard_fn(self.V, self.a)
        
        # Apply refractory period (lambda = 0)
        lam[is_ref] = 0.0
        
        p_sp = 1.0 - np.exp(-lam * dt)
        
        # 3. Spike draw
        U = np.random.uniform(0, 1, self.N)
        spikes = U < p_sp
        
        # Update spike times
        if np.any(spikes):
            self.last_spike_time[spikes] = t + dt # Spike happens at end of interval effectively
            
        return spikes, lam
