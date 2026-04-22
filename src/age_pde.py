import numpy as np

# Numba Optimization
try:
    from numba_kernels import step_pde_numba
    use_numba = True
except ImportError:
    use_numba = False

from pde_errors import PDESafetyError

class AgePDESolver:
    def __init__(self, t_len, dt, R_max, dr, ref_period, init_mode='delta0'):
        self.dt = dt
        self.dr = dr
        self.M = int(R_max / dr)
        self.r_grid = np.linspace(0, R_max, self.M + 1)
        self.init_mode = init_mode
        self.R_max = R_max 
        
        # Initial condition
        self.q = np.zeros(self.M + 1)
        
        if init_mode == 'delta0':
            # Option A: Delta at r=0 (Physical)
            # Distribute mass over first 10ms exactly to be numerically safe
            # but fundamentally close to 0.
            # 10ms spread
            spread_time = 0.010 
            k = max(1, int(spread_time / dr)) 
            self.q[:k] = 1.0
            # Normalize
            current_mass = np.sum(self.q) * self.dr
            self.q /= current_mass
            
            # Active tracking
            self.active_r = spread_time + self.dt * 2.0
            
        elif init_mode == 'warm_start':
            raise NotImplementedError("Warm start not yet implemented. Use delta0.")
            
        else:
            raise ValueError(f"Unknown or Forbidden init_mode: {init_mode}. Phase IX runs must use 'delta0'.")
        
        self.ref_period = ref_period
        
        # Initialize Buffer for Numba
        self.q_buffer = np.zeros_like(self.q)
        
        # Telemetry
        self.cum_leak = 0.0
        
    def step(self, rho_input):
        """
        rho_input: 
            - Scalar: Base hazard rate (clipped by refractory) [Phase I Legacy]
            - Vector (size M+1): Full hazard profile rho(r) [Phase II]
            
        Update q(r,t) -> q(r, t+dt)
        Returns A(t) (Firing rate)
        """
        if np.isscalar(rho_input):
            # Construct hazard profile: rho(r) = rho_base if r > ref else 0
            # Sliced construction?
            # We construct full rho_r because we might not know limit_idx here easily, or we can use limit_idx?
            # Let's just create full for now or optimize later. Full zeros is cheap.
            rho_r = np.zeros_like(self.q)
            mask_active = self.r_grid >= self.ref_period
            rho_r[mask_active] = rho_input
        else:
            # Vector input
            # If input is sliced, handle it.
            if len(rho_input) < len(self.q):
                # Padding
                # rho_r = np.zeros_like(self.q)
                # rho_r[:len(rho_input)] = rho_input
                # Efficient:
                rho_r = np.zeros_like(self.q)
                rho_r[:len(rho_input)] = rho_input
            elif len(rho_input) > len(self.q):
                rho_r = rho_input[:len(self.q)]
            else:
                rho_r = rho_input
            # Enforce refractory? usually encoded in profile, but safe to enforce
            mask_ref = self.r_grid < self.ref_period
            rho_r[mask_ref] = 0.0
            
        self.rho_last_used = rho_r.copy()
        
        # Compute A(t) = integral rho(r) q(r) dr
        # Using Riemann sum for consistency with Upwind
        density_hazard = rho_r * self.q
        A_t = np.sum(density_hazard) * self.dr
        
        # Active Set Update (include diffusion safety)
        self.active_r = min(self.R_max, self.active_r + self.dt * 1.2)
        limit_idx = int(self.active_r / self.dr) + 20
        limit_idx = min(limit_idx, self.M)
        
        # Explicit Upwind Scheme (Finite Volume Interpretation)
        if use_numba:
             # Use buffer to avoid allocation
             # self.q is q^n. q_buffer becomes q^{n+1}
             step_pde_numba(self.q, rho_r, self.dt, self.dr, A_t, self.q_buffer, limit_idx=limit_idx+1)
             # Swap: q now points to new data, q_buffer points to old data (to be overwritten next)
             self.q, self.q_buffer = self.q_buffer, self.q
        else:
             # Fallback
             q_new = np.zeros_like(self.q)
             # limit copy
             q_curr = self.q[1:limit_idx+1]
             q_prev = self.q[:limit_idx]
             
             courant = self.dt / self.dr
             rho_slice = rho_r[1:limit_idx+1]
             adv = q_curr - courant * (q_curr - q_prev)
             adv = np.maximum(adv, 0.0)
             q_new[1:limit_idx+1] = adv * np.exp(-self.dt * rho_slice)

             q0 = self.q[0] + courant * (A_t - self.q[0])
             q_new[0] = max(q0, 0.0) * np.exp(-self.dt * rho_r[0])
             self.q = q_new
        


        # Telemetry: Cumulative Leak (Flux out = q_end * v, v=1)
        self.cum_leak += self.q[-1] * self.dt
        
        return A_t
        
    def get_mass(self):
        return np.sum(self.q) * self.dr
        
    def get_diagnostics(self):
        """
        Return dictionary of diagnostic metrics.
        No longer raises PDESafetyError internally. Caller must handle checks.
        """
        mass = self.get_mass()
        
        # Tail Mass (Robust Truncation Check)
        # Check last 0.5s of age
        window = 0.5
        k = max(1, int(window / self.dr))
        tail_mass = np.sum(self.q[-k:]) * self.dr
        
        # Cumulative leak (Approximate flux out at boundary)
        # Flux out = q[end] * velocity (v=1)
        # To be accurate this should be integrated over time, but here we return instant
        # density at boundary as a proxy if needed, or the caller accumulates it.
        # But wait, user asked for "cum_leak += q[-1]*dt".
        # If I do it here, I need to store state.
        # Since get_diagnostics is called intermittently, I cannot accumulate here reliably 
        # unless I accumulate in `step`.
        # I didn't add cum_leak to __init__ yet. 
        # I will handle cum_leak in step() in a separate Edit or just return q[-1] here 
        # and let caller integrate? No, caller steps continuously but logs intermittently.
        # I MUST add cum_leak to __init__ and step.
        
        return {
            'mass': mass,
            'tail_mass': tail_mass,
            'cum_leak': self.cum_leak
        }
