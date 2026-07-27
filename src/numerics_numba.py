import numpy as np
from numba import njit

@njit
def seed_numba(seed):
    """Seed Numba's RNG independently of NumPy's Python-side generator."""
    np.random.seed(seed)

@njit(fastmath=True)
def compute_hazard_numba(V, a, theta_0, theta_1, theta_2, h_type_code):
    """
    Compute hazard function in place or return array.
    h_type_code: 0=exponential, 1=softplus
    """
    exponent = theta_0 + theta_1 * V + theta_2 * a

    # Stability clip
    exponent = np.clip(exponent, -50.0, 50.0)

    if h_type_code == 0: # exponential
        return np.exp(exponent)
    elif h_type_code == 1: # softplus
        return np.log(1.0 + np.exp(exponent))
    else:
        return np.exp(exponent)

@njit(fastmath=True)
def step_neuron_numba(V, a, last_spike_time, t, dt, u_t,
                      shared_noise, N,
                      tau_m, tau_a, kappa, sigma,
                      refractory,
                      theta_0, theta_1, theta_2, h_type_code,
                      noise_c_sqrt_shared, noise_c_sqrt_indep):
    """
    Single step of N neurons.
    V, a, last_spike_time: Arrays of shape (N,) modified in place.
    """
    sqrt_dt = np.sqrt(dt)

    # Generate independent noise here to avoid passing large array from Python
    # Numba supports np.random.standard_normal
    # Actually, generating N randoms might be faster in MKL/Numpy, but doing it here allows kernel fusion.
    # Let's try doing it loop-wise to avoid large allocation.

    spikes = np.zeros(N, dtype=np.bool_)
    lam = np.zeros(N, dtype=np.float64)

    # We can iterate, but let's stick to array ops which Numba handles well, is often autovectorized.
    # But to avoid large temp allocations, an explicit loop is often better in Numba.

    for i in range(N):
        xi_indep = np.random.standard_normal()
        xi = noise_c_sqrt_shared * shared_noise + noise_c_sqrt_indep * xi_indep

        # dV
        v_curr = V[i]
        a_curr = a[i]

        dV = (-v_curr / tau_m - a_curr + u_t) * dt + sigma * sqrt_dt * xi
        new_V = v_curr + dV

        # da
        da = (-a_curr / tau_a + kappa * v_curr) * dt
        new_a = a_curr + da

        # Update State
        V[i] = new_V
        a[i] = new_a

        # Refractory
        if (t - last_spike_time[i]) < refractory:
            lam[i] = 0.0
            lambda_val = 0.0
        else:
            exponent = theta_0 + theta_1 * new_V + theta_2 * new_a
            if exponent > 50.0: exponent = 50.0
            if exponent < -50.0: exponent = -50.0

            if h_type_code == 0:
                lambda_val = np.exp(exponent)
            else:
                lambda_val = np.log(1.0 + np.exp(exponent))

            lam[i] = lambda_val

        # Spike prob
        p_sp = 1.0 - np.exp(-lambda_val * dt)

        if np.random.random() < p_sp:
            spikes[i] = True
            last_spike_time[i] = t + dt

    return spikes, lam
