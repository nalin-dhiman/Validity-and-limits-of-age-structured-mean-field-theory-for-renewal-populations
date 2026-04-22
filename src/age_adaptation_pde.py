import numpy as np

# Numba Optimization
try:
    from numba import njit
    use_numba = True
except ImportError:
    use_numba = False

@njit
def step_density_upwind_split(q, dt, dr, hazard_rate, inflow, limit_idx=-1):
    """
    First-order upwind update for the density with an exact hazard sink.

    The hazard contribution is integrated as q <- q * exp(-rho dt), which keeps the
    density non-negative even when rho dt becomes large.
    """
    M = len(q) - 1
    q_new = np.zeros_like(q)
    courant = dt / dr

    eff_limit = M
    if limit_idx > 0 and limit_idx < M:
        eff_limit = limit_idx

    for i in range(1, eff_limit + 1):
        adv = q[i] - courant * (q[i] - q[i - 1])
        if adv < 0.0:
            adv = 0.0
        q_new[i] = adv * np.exp(-hazard_rate[i] * dt)

    q0 = q[0] + courant * (inflow - q[0])
    if q0 < 0.0:
        q0 = 0.0
    q_new[0] = q0 * np.exp(-hazard_rate[0] * dt)

    return q_new

@njit
def step_field_upwind_split(y, dt, dr, source_term, linear_decay, hazard_rate, inflow, limit_idx=-1):
    """
    Upwind/source step with semi-implicit linear decay followed by an exact hazard sink.

    This keeps the stiff hazard term from destabilizing the transport step while preserving
    the original first-order accuracy of the solver.
    """
    M = len(y) - 1
    y_new = np.zeros_like(y)
    courant = dt / dr

    eff_limit = M
    if limit_idx > 0 and limit_idx < M:
        eff_limit = limit_idx

    for i in range(1, eff_limit + 1):
        adv_src = y[i] - courant * (y[i] - y[i - 1]) + dt * source_term[i]
        adv_src = adv_src / (1.0 + dt * linear_decay[i])
        y_new[i] = adv_src * np.exp(-hazard_rate[i] * dt)

    y0 = y[0] + courant * (inflow - y[0]) + dt * source_term[0]
    y0 = y0 / (1.0 + dt * linear_decay[0])
    y_new[0] = y0 * np.exp(-hazard_rate[0] * dt)

    return y_new

class AgeAdaptationPDESolver:
    def __init__(self, t_len, dt, R_max, dr, params, use_jensen=True, init_mode='delta0'):
        self.dt = dt
        self.dr = dr
        self.M = int(R_max / dr)
        self.r_grid = np.linspace(0, R_max, self.M + 1)
        self.use_jensen = use_jensen
        self.R_max = R_max
        self.init_mode = init_mode
        
        # Params
        p = params['neuron']
        self.tau_m = p['tau_m']
        self.tau_a = p['tau_a']
        self.kappa = p['kappa']
        self.sigma = p['sigma'] # Intrinsic noise
        
        sg = params['spike_gen']
        self.refractory = sg['refractory']
        self.theta_0 = sg['theta_0']
        self.theta_1 = sg['theta_1']
        self.theta_2 = sg['theta_2']
        self.type = sg.get('type', 'exponential')
        self.use_jensen_requested = bool(use_jensen)
        self.use_jensen = bool(use_jensen and self.type == 'exponential')

        # State Arrays (Conservative Variables)
        # q: Density of age
        # M_field: q * m
        # V_field: q * v
        
        # In this formulation, we evolve (q, M, V) directly.
        # Recover intensive m, v for hazard computation.
        
        # Initialization
        self.q = np.zeros(self.M + 1)
        
        if init_mode == 'delta0':
            # Option A: Delta at r=0 (Physical)
            # 10ms spread
            spread_time = 0.010
            k = max(1, int(spread_time / dr)) 
            self.q[:k] = 1.0
            # Normalize
            current_mass = np.sum(self.q) * self.dr
            self.q /= current_mass

            # Phase IX Optimization: Track Active Front
            # Initialize active_r to cover the initial spread
            self.active_r = spread_time + self.dt * 2.0 # Margin
            
        elif init_mode == 'warm_start':
             raise NotImplementedError("Warm start not yet implemented.")
             
        else:
            raise ValueError(f"Unknown or Forbidden init_mode: {init_mode}. Phase IX runs must use 'delta0'.")
        
        # M, V init to 0 (rest)
        self.M_field = np.zeros_like(self.q)
        self.V_field = np.zeros_like(self.q)
        
        # Precompute Variance Profile for Jensen
        # Var(V|s) approx (sigma^2 * tau_m / 2) * (1 - exp(-2s/tau_m))
        self.var_v_profile = (self.sigma**2 * self.tau_m / 2.0) * \
                             (1.0 - np.exp(-2.0 * self.r_grid / self.tau_m))

        # Telemetry
        self.cum_leak = 0.0

    def compute_hazard(self, v_arr, m_arr):
        """
        Compute hazard rate rho(s).
        """
        exponent = self.theta_0 + self.theta_1 * v_arr + self.theta_2 * m_arr
        
        # Jensen Corrections
        if self.use_jensen:
            # Phase IX Optimization: Slice var profile if input is sliced
            if len(v_arr) < len(self.var_v_profile):
                var_slice = self.var_v_profile[:len(v_arr)]
                exponent += 0.5 * (self.theta_1**2) * var_slice
            else:
                exponent += 0.5 * (self.theta_1**2) * self.var_v_profile

        exponent = np.clip(exponent, -50, 50)

        if self.type == 'exponential':
            rho = np.exp(exponent)
        elif self.type == 'softplus':
            rho = np.maximum(exponent, 0.0) + np.log1p(np.exp(-np.abs(exponent)))
        else:
            raise ValueError(f"Unknown hazard type: {self.type}")
            
        # Refractory
        # Check size of rho vs r_grid
        N_rho = len(rho)
        if N_rho < len(self.r_grid):
            # Only check up to N_rho
            # Efficiently: r_grid starts at 0, increases.
            # But safer to slice.
            mask_ref = self.r_grid[:N_rho] < self.refractory
        else:
            mask_ref = self.r_grid < self.refractory
            
        rho[mask_ref] = 0.0
        
        return rho
        
    def step(self, u_t):
        """
        Evolve state by dt.
        u_t: scalar input
        """
        eps = 1e-12
        
        # Phase IX Optimization: Active Set Tracking
        # Update active_r (Physically moves at speed 1, but diffusion spreads it)
        # Use 1.2x speed to outrun numerical diffusion
        self.active_r = min(self.R_max, self.active_r + self.dt * 1.2)
        
        # Calculate limit_idx with safety margin
        # margin of 20 cells is plenty for numerical diffusion control
        limit_idx = int(self.active_r / self.dr) + 20
        limit_idx = min(limit_idx, self.M)
        
        # 1. Recover Intensive Variables
        # m = M / q
        # v = V / q
        # Optimization: Only recover up to limit_idx
        
        # Slices for active region
        q_slice = self.q[:limit_idx+1]
        M_slice = self.M_field[:limit_idx+1]
        V_slice = self.V_field[:limit_idx+1]
        
        mask = q_slice > eps
        m_curr = np.zeros_like(q_slice)
        v_curr = np.zeros_like(q_slice)
        
        m_curr[mask] = M_slice[mask] / q_slice[mask]
        v_curr[mask] = V_slice[mask] / q_slice[mask]
        
        # 2. Compute Hazard Field (Optimized Sliced)
        rho_slice = self.compute_hazard(v_curr, m_curr)
        
        # 3. Compute Activity A(t)
        density_hazard = rho_slice * q_slice
        A_t = np.sum(density_hazard) * self.dr
        
        # 4. Boundary Values (Reinjection)
        # q_0_in = A(t)
        # M_0_in = A(t) * m_reset
        # V_0_in = A(t) * v_reset
        
        # Reset Logic
        if A_t > eps:
            num_v = np.sum(rho_slice * V_slice) * self.dr
            num_m = np.sum(rho_slice * M_slice) * self.dr
            v_reset = num_v / A_t
            m_reset = num_m / A_t
        else:
            v_reset = 0.0 # Rest
            m_reset = 0.0
            
        # 5. Evolve (Conservative Form)
        
        # 5a. q
        # dq/dt + dq/ds = -rho * q
        # Source=0, Decay=rho
        # We pass full arrays but limit loop
        # We need to construct source/decay arrays that are safe to access up to limit_idx.
        # simple: pass a zero array of sufficient size? 
        # Actually `step_advection_conservative` takes `source_term` and `decay_rate`.
        # For q: Source=0 (zeros_like(q)), Decay=rho.
        # But rho is `rho_slice`. Length `limit_idx+1`.
        # `step_advection_conservative` accesses `decay_rate[i]`. i goes to `limit_idx`.
        # `rho_slice` has index up to `limit_idx`. So accessing `rho_slice[limit_idx]` is valid.
        # But we need to pass `rho_slice`? 
        # `step_advection_conservative` expects `decay_rate` to be array.
        # If we pass `rho_slice`, it works for relevant indices.
        # But `source_term`? `np.zeros_like(self.q)` is expensive (full alloc).
        # We can pass `np.zeros(limit_idx+1)`!
        
        q_new = step_density_upwind_split(
            self.q,
            self.dt,
            self.dr,
            rho_slice,
            A_t,
            limit_idx=limit_idx
        )
        
        # 5b. M = qm
        # Source = kappa * V
        # Decay = rho + 1/tau_a
        M_source_slice = self.kappa * V_slice
        M_linear_decay = np.full(limit_idx + 1, 1.0 / self.tau_a)
        M_inflow = A_t * m_reset

        M_new = step_field_upwind_split(
            self.M_field,
            self.dt,
            self.dr,
            source_term=M_source_slice,
            linear_decay=M_linear_decay,
            hazard_rate=rho_slice,
            inflow=M_inflow,
            limit_idx=limit_idx
        )
        
        # 5c. V = qv
        # Source = u*q - M
        # Decay = rho + 1/tau_m
        V_source_slice = u_t * q_slice - M_slice
        V_linear_decay = np.full(limit_idx + 1, 1.0 / self.tau_m)
        V_inflow = A_t * v_reset

        V_new = step_field_upwind_split(
            self.V_field,
            self.dt,
            self.dr,
            source_term=V_source_slice,
            linear_decay=V_linear_decay,
            hazard_rate=rho_slice,
            inflow=V_inflow,
            limit_idx=limit_idx
        )
        
        # Commit
        self.q = q_new
        self.M_field = M_new
        self.V_field = V_new
        
        # Diagnostic accessors for external use
        # Just store the computed m_curr, v_curr (pad with nan or 0?)
        # For compatibility, we should return full size m, v or update them.
        # But `self.m` and `self.v` are just for visualization usually.
        # Let's store full size zeros and update active part?
        self.m = np.zeros_like(self.q)
        self.m[:limit_idx+1] = m_curr
        
        self.v = np.zeros_like(self.q)
        self.v[:limit_idx+1] = v_curr
        
        # Telemetry
        self.cum_leak += self.q[-1] * self.dt
        
        return A_t
        
    def get_diagnostics(self):
        """
        Return dictionary of diagnostic metrics.
        No longer raises PDESafetyError internally.
        """
        mass = np.sum(self.q) * self.dr
        
        # Tail Mass (Robust Truncation Check)
        window = 0.5
        k = max(1, int(window / self.dr))
        tail_mass = np.sum(self.q[-k:]) * self.dr
        
        return {
            'mass': mass,
            'tail_mass': tail_mass,
            'cum_leak': self.cum_leak
        }

        
