import numpy as np

class HazardFunction:
    def __init__(self, params):
        self.params = params
        self.theta_0 = params['spike_gen']['theta_0']
        self.theta_1 = params['spike_gen']['theta_1']
        self.theta_2 = params['spike_gen']['theta_2']
        self.type = params['spike_gen'].get('type', 'exponential')
        
    def __call__(self, V, a):
        exponent = self.theta_0 + self.theta_1 * V + self.theta_2 * a
        # Stability clip
        exponent = np.clip(exponent, -50, 50)
        
        if self.type == 'exponential':
            return np.exp(exponent)
        elif self.type == 'softplus':
            return np.maximum(exponent, 0.0) + np.log1p(np.exp(-np.abs(exponent)))
        else:
            raise ValueError(f"Unknown hazard type: {self.type}")
