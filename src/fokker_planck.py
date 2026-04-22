import numpy as np

class FokkerPlanckSolver:
    def __init__(self, v_grid, dt, tau_m, mu, sigma, hazard_fn):
        self.v = v_grid
        self.dv = v_grid[1] - v_grid[0]
        self.dt = dt
        self.tau_m = tau_m
        self.mu = mu
        self.sigma = sigma
        self.hazard_fn = hazard_fn
        
        self.p = np.zeros_like(v_grid)
        # Initialize Gaussian
        self.p = np.exp(-(self.v - mu)**2 / (2*0.1**2))
        self.p /= np.sum(self.p) * self.dv
        
    def step(self):
        # Operator splitting: Transport + Diffusion + Hazard
        # 1. Transport (Advection)
        # d(vp)/dv approx (vp)_{i+1} - (vp)_i ... Upwind depends on sign of velocity
        # Velocity = (mu - v)/tau_m
        
        vel = (self.mu - self.v) / self.tau_m
        
        # Upwind flux
        # Flux F = vel * p
        # If vel > 0, F_i = vel_i * p_i
        # If vel < 0, F_i = vel_i * p_{i+1}
        
        F = np.zeros_like(self.p)
        pos_v = vel > 0
        neg_v = ~pos_v
        
        # p[i] corresponds to center? Use edges for Flux?
        # Simple finite difference on cell centers for simplicity
        # F_{i+1/2} 
        
        p = self.p
        
        # Explicit Upwind
        # dp/dt = - dF/dv
        flux_out = np.zeros_like(p)
        flux_in = np.zeros_like(p)
        
        # Positive velocity: moves i -> i+1
        # Flux leaving i: p[i]*vel[i]
        # Flux entering i: p[i-1]*vel[i-1]
        
        # Negative velocity: moves i+1 -> i
        # Flux leaving i: p[i]*|vel[i]| -> i-1
        # Flux entering i: p[i+1]*|vel[i+1]|
        
        # Actually easier to just assemble F at edges
        # Just use standard upwind
        
        dp_adv = np.zeros_like(p)
        
        # Rightward flow (vel > 0)
        # p[i] += -dt/dx * (vel[i]*p[i] - vel[i-1]*p[i-1])
        shift_right = np.roll(p * vel, 1)
        shift_right[0] = 0 # Boundary
        adv_right = -(vel * p - shift_right) / self.dv
        
        # Leftward flow (vel < 0)
        # p[i] += -dt/dx * (vel[i]*p[i] - vel[i+1]*p[i+1]) # vel is negative, so careful
        # Drift term is - d(v p)/dx
        
        # Let's use Chang-Cooper? Or simple semi-implicit?
        # For this task: Simple explicit is fine if dt is small.
        # sigma^2/2 * d2p/dv2
        
        D = 0.5 * self.sigma**2
        
        # Diffusion (Central)
        d2p = (np.roll(p, 1) - 2*p + np.roll(p, -1)) / (self.dv**2)
        # Handle boundaries (No flux or Absorbing?)
        # Here boundaries are "truncation" of domain [Vmin, Vmax].
        # Assume Vmin/Vmax far away -> p=0
        d2p[0] = 0; d2p[-1] = 0
        
        # Hazard Sink
        # dp/dt = - hazard * p
        # rho(v)
        rho = self.hazard_fn(self.v, 0.0) # a=0 for this test
        # Clamp rho for stability of explicit Euler
        # dt=1e-5. Max stable rho < 1/dt = 1e5.
        rho = np.minimum(rho, 50000.0)
        
        rate_t = np.sum(rho * p) * self.dv
        
        # Update
        # Combine
        # dp = (Advection + Diffusion - Hazard) * dt
        # Re-implementation of advection for clarity:
        flux = vel * p
        # Gradient of flux using upwind
        d_flux = np.zeros_like(p)
        
        # Loop slow but safe
        for i in range(1, len(p)-1):
            f_right = flux[i] if vel[i] > 0 else flux[i+1] # Flux at i+1/2
            f_left = flux[i-1] if vel[i-1] > 0 else flux[i] # Flux at i-1/2
            d_flux[i] = (f_right - f_left) / self.dv
            
        dp = (-d_flux + D * d2p - rho * p) * self.dt
        
        self.p += dp
        self.p = np.maximum(self.p, 0)
        
        # Renormalize? 
        # Mass loss due to hazard is Rate.
        # But this is NOT Age-PDE where A(t) re-enters at age 0.
        # This is V-space FP.
        # A(t) re-enters at V_reset?
        # Prompt says "absorbing boundary and reinjection".
        # But here we use Soft Hazard.
        # So "Death" at v implies "Birth" at V_reset.
        
        # Reinjection
        # Find index of V_reset (approx 0)
        idx_reset = np.argmin(np.abs(self.v - 0.0))
        self.p[idx_reset] += rate_t * self.dt / self.dv
        
        return rate_t
