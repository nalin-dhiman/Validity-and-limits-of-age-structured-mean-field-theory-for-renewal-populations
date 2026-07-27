import numpy as np

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
            # Approximate an age-zero delta over the first 10 ms.
            spread_time = 0.010
            k = max(1, int(spread_time / dr))
            self.q[:k] = 1.0
            current_mass = np.sum(self.q) * self.dr
            self.q /= current_mass

            self.active_r = spread_time + self.dt * 2.0

        elif init_mode == 'warm_start':
            raise NotImplementedError("Warm start not yet implemented. Use delta0.")

        else:
            raise ValueError(f"unsupported init_mode: {init_mode}")

        self.ref_period = ref_period

        self.cum_leak = 0.0

    def step(self, rho_input):
        """Advance the age density and return the activity in hertz.

        ``rho_input`` may be a scalar active-age hazard or an age-resolved
        vector. The absolute refractory interval is enforced in either case.
        """
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

        self.active_r = min(self.R_max, self.active_r + self.dt)
        # The full finite domain is advanced.  Truncating at a moving active
        # front turns numerical diffusion at that front into spurious mass loss.
        limit_idx = self.M
        courant = self.dt / self.dr
        if not (0.0 < courant <= 1.0):
            raise ValueError(f"upwind CFL requires 0 < dt/dr <= 1; got {courant}")

        q = self.q[:limit_idx + 1]
        q_surv = q * np.exp(-rho_r[:limit_idx + 1] * self.dt)
        loss = float(np.sum(q - q_surv) * self.dr)
        A_t = loss / self.dt

        q_new = np.zeros_like(self.q)
        q_new[0] = q_surv[0] - courant * (q_surv[0] - A_t)
        if limit_idx >= 1:
            q_new[1:limit_idx + 1] = q_surv[1:] - courant * np.diff(q_surv)
        q_new[:limit_idx + 1] = np.maximum(q_new[:limit_idx + 1], 0.0)
        self.cum_leak += q_surv[-1] * self.dt if limit_idx == self.M else 0.0
        self.q = q_new

        return A_t

    def get_mass(self):
        return np.sum(self.q) * self.dr

    def get_diagnostics(self):
        """Return mass, tail-mass, and accumulated boundary-leak diagnostics."""
        mass = self.get_mass()

        window = 0.5
        k = max(1, int(window / self.dr))
        tail_mass = np.sum(self.q[-k:]) * self.dr

        return {
            'mass': mass,
            'tail_mass': tail_mass,
            'cum_leak': self.cum_leak
        }
