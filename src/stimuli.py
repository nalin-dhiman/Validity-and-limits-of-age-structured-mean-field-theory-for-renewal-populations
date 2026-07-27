import numpy as np

class StimulusGenerator:
    def __init__(self, dt):
        self.dt = dt

    def generate_ou(self, T, tau_c, sigma_u, mu_u=0.0, seed=None):
        """
        Generate Ornstein-Uhlenbeck process.
        du = -u/tau_c * dt + sigma_u * dW
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        n_steps = int(T / self.dt)
        u = np.zeros(n_steps)
        u[0] = mu_u # Start at mean

        # Precompute constants
        drift = -1.0 / tau_c
        diff = sigma_u
        sqrt_dt = np.sqrt(self.dt)

        noise = rng.normal(0, 1, n_steps)

        for t in range(n_steps - 1):
            # Euler-Maruyama
            # Note: u[t] in the drift term refers to deviation from 0 if it's centered OU.
            # If mu_u is the mean, the equation is d(u - mu) = -(u - mu)/tau dt + ...
            # Implementation: generate centered OU x, then return x + mu
            # Doing centered generation here:
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

        # Spectrum: Flat up to f_c, zero elsewhere (or smooth cutoff)
        # Using hard cutoff as implied by name, or a specific profile.
        # "Generate in Fourier domain with cutoff f_c"

        spectrum = np.zeros(n_steps, dtype=complex)
        mask = np.abs(freqs) <= f_c

        # Random phases
        phases = np.random.uniform(0, 2*np.pi, n_steps)

        # Magnitude: constant in band (White noise in band)
        # Adjusted so that Inverse FFT has roughly unit variance before scaling
        spectrum[mask] = np.exp(1j * phases[mask])

        # Enforce Hermitian symmetry for real output
        # spectrum[-k] = conj(spectrum[k])
        # This is automatically handled if we use rfft/irfft or construct carefully.
        # Easier to use: generate white noise in time, FFT, filter, IFFT.

        white_noise = np.random.normal(0, 1, n_steps)
        ft = np.fft.fft(white_noise)
        ft[~mask] = 0
        u_raw = np.fft.ifft(ft).real

        # Normalize variance
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

        # We need continuity.
        # Generate part 1
        u_pre = self.generate_ou(t_switch, **params_pre, seed=seed) # Note: seed used here

        # Generate part 2 step-by-step to maintain state
        u_post = np.zeros(n_post)
        curr_u = u_pre[-1]

        tau_c = params_post['tau_c']
        sigma_u = params_post['sigma_u']
        mu_u = params_post.get('mu_u', 0.0)

        drift = -1.0 / tau_c
        sqrt_dt = np.sqrt(self.dt)

        # If we re-seed for second part, we break the single stream.
        # But we used seed for pre.
        # Should rely on numpy global state if seed is None, or carefully manage random stream.
        # Implementation assumes seed sets state once at start.
        if seed is not None:
            # We already used the seed for u_pre. Validating this flow:
            # generate_ou sets seed. If we call it, it resets.
            # Better to not reset inside the sub-call if we want continuity of random stream?
            # Or just set seed at start of THIS function.
            # But generate_ou resets seed.
            pass

        # Manual stepping for post to ensure continuity
        noise = np.random.normal(0, 1, n_post)

        for t in range(n_post):
            curr_val = curr_u - mu_u
            dx = (drift * curr_val) * self.dt + sigma_u * sqrt_dt * noise[t]
            curr_u = (curr_val + dx) + mu_u
            u_post[t] = curr_u

        return np.concatenate([u_pre, u_post])
