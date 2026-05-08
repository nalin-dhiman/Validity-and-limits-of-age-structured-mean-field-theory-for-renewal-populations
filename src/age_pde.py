import numpy as np

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
        
        self.q = np.zeros(self.M + 1)
        
        if init_mode == 'delta0':
            
            spread_time = 0.010 
            k = max(1, int(spread_time / dr)) 
            self.q[:k] = 1.0
            current_mass = np.sum(self.q) * self.dr
            self.q /= current_mass
            
            self.active_r = spread_time + self.dt * 2.0
            
        elif init_mode == 'warm_start':
            raise NotImplementedError("Warm start not yet implemented. Use delta0.")
            
        else:
            raise ValueError(f"Unknown or Forbidden init_mode: {init_mode}. Phase IX runs must use 'delta0'.")
        
        self.ref_period = ref_period
        
        self.q_buffer = np.zeros_like(self.q)
        
        self.cum_leak = 0.0
        
    def step(self, rho_input):
     
        if np.isscalar(rho_input):
           
            rho_r = np.zeros_like(self.q)
            mask_active = self.r_grid >= self.ref_period
            rho_r[mask_active] = rho_input
        else:
           
            if len(rho_input) < len(self.q):
            
                rho_r = np.zeros_like(self.q)
                rho_r[:len(rho_input)] = rho_input
            elif len(rho_input) > len(self.q):
                rho_r = rho_input[:len(self.q)]
            else:
                rho_r = rho_input
            mask_ref = self.r_grid < self.ref_period
            rho_r[mask_ref] = 0.0
            
        self.rho_last_used = rho_r.copy()
        

        density_hazard = rho_r * self.q
        A_t = np.sum(density_hazard) * self.dr
        
        self.active_r = min(self.R_max, self.active_r + self.dt * 1.2)
        limit_idx = int(self.active_r / self.dr) + 20
        limit_idx = min(limit_idx, self.M)
        
        if use_numba:
             
             step_pde_numba(self.q, rho_r, self.dt, self.dr, A_t, self.q_buffer, limit_idx=limit_idx+1)
             self.q, self.q_buffer = self.q_buffer, self.q
        else:
             q_new = np.zeros_like(self.q)
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
        


        self.cum_leak += self.q[-1] * self.dt
        
        return A_t
        
    def get_mass(self):
        return np.sum(self.q) * self.dr
        
    def get_diagnostics(self):
       
        mass = self.get_mass()
        
    
        window = 0.5
        k = max(1, int(window / self.dr))
        tail_mass = np.sum(self.q[-k:]) * self.dr
        
        
        return {
            'mass': mass,
            'tail_mass': tail_mass,
            'cum_leak': self.cum_leak
        }
