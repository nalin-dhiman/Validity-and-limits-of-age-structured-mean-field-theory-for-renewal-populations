import numpy as np

try:
    from numba import njit
    use_numba = True
except ImportError:
    use_numba = False

@njit
def step_density_upwind_split(q, dt, dr, hazard_rate, inflow, limit_idx=-1):
   
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
        
        p = params['neuron']
        self.tau_m = p['tau_m']
        self.tau_a = p['tau_a']
        self.kappa = p['kappa']
        self.sigma = p['sigma']
        
        sg = params['spike_gen']
        self.refractory = sg['refractory']
        self.theta_0 = sg['theta_0']
        self.theta_1 = sg['theta_1']
        self.theta_2 = sg['theta_2']
        self.type = sg.get('type', 'exponential')
        self.use_jensen_requested = bool(use_jensen)
        self.use_jensen = bool(use_jensen and self.type == 'exponential')

       
        self.q = np.zeros(self.M + 1)
        
        if init_mode == 'delta0':
         
            spread_time = 0.010
            k = max(1, int(spread_time / dr)) 
            self.q[:k] = 1.0
            current_mass = np.sum(self.q) * self.dr
            self.q /= current_mass

           
            self.active_r = spread_time + self.dt * 2.0 
            
        elif init_mode == 'warm_start':
             raise NotImplementedError("Warm start not yet implemented.")
             
        else:
            raise ValueError(f"Unknown or Forbidden init_mode: {init_mode}. Phase IX runs must use 'delta0'.")
        
        self.M_field = np.zeros_like(self.q)
        self.V_field = np.zeros_like(self.q)
       
        self.var_v_profile = (self.sigma**2 * self.tau_m / 2.0) * \
                             (1.0 - np.exp(-2.0 * self.r_grid / self.tau_m))

        self.cum_leak = 0.0

    def compute_hazard(self, v_arr, m_arr):
       
        exponent = self.theta_0 + self.theta_1 * v_arr + self.theta_2 * m_arr
        
        if self.use_jensen:
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
            
        
        N_rho = len(rho)
        if N_rho < len(self.r_grid):
            
            mask_ref = self.r_grid[:N_rho] < self.refractory
        else:
            mask_ref = self.r_grid < self.refractory
            
        rho[mask_ref] = 0.0
        
        return rho
        
    def step(self, u_t):
       
        eps = 1e-12
        
        
        self.active_r = min(self.R_max, self.active_r + self.dt * 1.2)
        
       
        limit_idx = int(self.active_r / self.dr) + 20
        limit_idx = min(limit_idx, self.M)
        
       
        q_slice = self.q[:limit_idx+1]
        M_slice = self.M_field[:limit_idx+1]
        V_slice = self.V_field[:limit_idx+1]
        
        mask = q_slice > eps
        m_curr = np.zeros_like(q_slice)
        v_curr = np.zeros_like(q_slice)
        
        m_curr[mask] = M_slice[mask] / q_slice[mask]
        v_curr[mask] = V_slice[mask] / q_slice[mask]
        
        rho_slice = self.compute_hazard(v_curr, m_curr)
        
        density_hazard = rho_slice * q_slice
        A_t = np.sum(density_hazard) * self.dr
        
       
        if A_t > eps:
            num_v = np.sum(rho_slice * V_slice) * self.dr
            num_m = np.sum(rho_slice * M_slice) * self.dr
            v_reset = num_v / A_t
            m_reset = num_m / A_t
        else:
            v_reset = 0.0 
            m_reset = 0.0
            
        
        q_new = step_density_upwind_split(
            self.q,
            self.dt,
            self.dr,
            rho_slice,
            A_t,
            limit_idx=limit_idx
        )
        
       
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
        
        self.q = q_new
        self.M_field = M_new
        self.V_field = V_new
        
        
        self.m = np.zeros_like(self.q)
        self.m[:limit_idx+1] = m_curr
        
        self.v = np.zeros_like(self.q)
        self.v[:limit_idx+1] = v_curr
        
        self.cum_leak += self.q[-1] * self.dt
        
        return A_t
        
    def get_diagnostics(self):
   
        mass = np.sum(self.q) * self.dr
        
        window = 0.5
        k = max(1, int(window / self.dr))
        tail_mass = np.sum(self.q[-k:]) * self.dr
        
        return {
            'mass': mass,
            'tail_mass': tail_mass,
            'cum_leak': self.cum_leak
        }

        
