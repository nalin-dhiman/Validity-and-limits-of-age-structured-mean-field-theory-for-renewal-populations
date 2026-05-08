import numpy as np

class StimulusGenerator:
    def __init__(self, dt):
        self.dt = dt

    def generate_ou(self, T, tau_c, sigma_u, mu_u=0.0, seed=None):
        """
        Generate Ornstein-Uhlenbeck process.
        du = -u/tau_c * dt + sigma_u * dW
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_steps = int(T / self.dt)
        u = np.zeros(n_steps)
        u[0] = mu_u # Start at mean
        
        drift = -1.0 / tau_c
        diff = sigma_u
        sqrt_dt = np.sqrt(self.dt)
        
        noise = np.random.normal(0, 1, n_steps)
        
        for t in range(n_steps - 1):
           
            curr = u[t] - mu_u
            dx = (drift * curr) * self.dt + diff * sqrt_dt * noise[t]
            u[t+1] = (curr + dx) + mu_u
            
        return u

    def generate_band_limited(self, T, f_c, sigma_u, mu_u=0.0, seed=None):
        """
        Generate Band-Limited Gaussian Noise via FFT.
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_steps = int(T / self.dt)
        freqs = np.fft.fftfreq(n_steps, d=self.dt)
        
       
        
        spectrum = np.zeros(n_steps, dtype=complex)
        mask = np.abs(freqs) <= f_c
        
        phases = np.random.uniform(0, 2*np.pi, n_steps)
        
        
        spectrum[mask] = np.exp(1j * phases[mask])
        
        
        
        white_noise = np.random.normal(0, 1, n_steps)
        ft = np.fft.fft(white_noise)
        ft[~mask] = 0
        u_raw = np.fft.ifft(ft).real
        
        u_raw = u_raw / np.std(u_raw)
        
        return u_raw * sigma_u + mu_u

    def generate_non_stationary(self, T, params_pre, params_post, t_switch=None, seed=None):
        """
        Switch parameters at T/2.
        Supports switching tau_c or sigma_u for OU process.
        """
        if seed is not None:
            np.random.seed(seed)
            
        if t_switch is None:
            t_switch = T / 2.0
            
        n_pre = int(t_switch / self.dt)
        n_total = int(T / self.dt)
        n_post = n_total - n_pre
        
       
        u_pre = self.generate_ou(t_switch, **params_pre, seed=seed) 
        
        u_post = np.zeros(n_post)
        curr_u = u_pre[-1]
        
        tau_c = params_post['tau_c']
        sigma_u = params_post['sigma_u']
        mu_u = params_post.get('mu_u', 0.0)
        
        drift = -1.0 / tau_c
        sqrt_dt = np.sqrt(self.dt)
        
       
        if seed is not None:
           
            pass

        noise = np.random.normal(0, 1, n_post)
        
        for t in range(n_post):
            curr_val = curr_u - mu_u
            dx = (drift * curr_val) * self.dt + sigma_u * sqrt_dt * noise[t]
            curr_u = (curr_val + dx) + mu_u
            u_post[t] = curr_u
            
        return np.concatenate([u_pre, u_post])

