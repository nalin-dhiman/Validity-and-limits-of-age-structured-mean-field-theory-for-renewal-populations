import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def step_neuron_numba(V, a, last_spike_time, dt, t, u_ext, sigma, kappa, tau_m, tau_a, refractory, theta_0, theta_1, theta_2, noise_xi):
    """
    Numba optimized step for Exponential Neuron.
    Updates V, a in-place.
    Returns: spikes (boolean array)
    """
    N = V.shape[0]
    sqrt_dt = np.sqrt(dt)
    
    # Dynamics
    # dV = (-V/tau_m - a + u)*dt + sigma*sqrt(dt)*xi
    # da = (-a/tau_a + kappa*V)*dt
    
    # Vectorized ops are supported in nopython
    dV = (-V / tau_m - a + u_ext) * dt + sigma * sqrt_dt * noise_xi
    da = (-a / tau_a + kappa * V) * dt
    
    V[:] += dV
    a[:] += da
    
    # Hazard
    # lam = exp(th0 + th1*V + th2*a)
    exponent = theta_0 + theta_1 * V + theta_2 * a
    # clip exponent to avoid overflow
    # exponent = np.clip(exponent, -50.0, 50.0)
    # Numba clip?
    for i in range(N):
        if exponent[i] > 50.0: exponent[i] = 50.0
        if exponent[i] < -50.0: exponent[i] = -50.0
        
    lam = np.exp(exponent)
    
    # Refractory
    # is_ref = (t - last_spike) < ref
    # lam[is_ref] = 0.0
    for i in range(N):
        if (t - last_spike_time[i]) < refractory:
            lam[i] = 0.0
            
    p_sp = 1.0 - np.exp(-lam * dt)
    
    # Spikes
    # U = rand check is outside? Or we generate one rand per neuron here?
    # We can use np.random.rand() inside?
    # Or pass seeds?
    # If we use np.random.rand(N) inside, it uses numba random.
    spikes = np.zeros(N, dtype=np.bool_)
    
    for i in range(N):
        if np.random.random() < p_sp[i]:
            spikes[i] = True
            last_spike_time[i] = t + dt
            
    return spikes, lam

@jit(nopython=True, cache=True)
def step_pde_numba(q, rho_r, dt, dr, A_t, q_out, limit_idx=-1):
    """
    Finite Volume Step for Age PDE.
    q: Density array (N_r,)
    rho_r: Hazard array (N_r,)
    A_t: Boundary flux
    q_out: Output buffer (must be same shape as q)
    """
    N_r = q.shape[0]
    courant = dt / dr
    
    limit = N_r
    if limit_idx > 0 and limit_idx < N_r:
        limit = limit_idx
        
    # Interior 1..limit-1
    # We want to update up to index limit-1? 
    # If limit comes from limit_idx (which is an index), we want to update THAT index too to capture flux.
    # So we should Loop 1..limit (inclusive) -> range(1, limit+1)
    # BUT we need bound check.
    
    # Logic fix:
    # If limit_idx was passed (e.g. 30), we want to update index 30.
    # range(1, 31).
    # If no limit passed, limit=N_r. range(1, N_r) updates N_r-1. Correct.
    # So if limit_idx passed, we want range(1, limit_idx + 1).
    # My previous code set limit = limit_idx.
    # range(1, limit) -> 1..limit_idx-1. MISSES limit_idx.
    
    # Adjusted Logic:
    effective_limit = N_r
    if limit_idx > 0 and limit_idx < N_r:
        effective_limit = limit_idx + 1 # +1 to include the boundary cell in update
        # Check safety (limit_idx+1 must be < N_r? NO, limit_idx < N_r allowed)
        # If limit_idx = N_r-1 (last cell), limit = N_r. range(1, N_r) OK.
    
    for i in range(1, effective_limit):
        adv = q[i] - courant * (q[i] - q[i-1])
        if adv < 0.0:
            adv = 0.0
        q_out[i] = adv * np.exp(-dt * rho_r[i])
        
    # Boundary i=0
    q0 = q[0] + courant * (A_t - q[0])
    if q0 < 0.0:
        q0 = 0.0
    q_out[0] = q0 * np.exp(-dt * rho_r[0])
    
    return q_out
