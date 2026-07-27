import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(function):
        """Fallback decorator when Numba is unavailable."""
        return function

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

        # Conservative fields q, q E[a], and q E[V].
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
            raise NotImplementedError("Warm start not yet implemented.")

        else:
            raise ValueError(f"unsupported init_mode: {init_mode}")

        self.M_field = np.zeros_like(self.q)
        self.V_field = np.zeros_like(self.q)

        # Prescribed conditional voltage variance used by the Jensen closure.
        self.var_v_profile = (self.sigma**2 * self.tau_m / 2.0) * \
                             (1.0 - np.exp(-2.0 * self.r_grid / self.tau_m))

        # Telemetry
        self.cum_leak = 0.0

    def compute_hazard(self, v_arr, m_arr):
        """Compute the age-resolved escape hazard."""
        exponent = self.theta_0 + self.theta_1 * v_arr + self.theta_2 * m_arr

        # Jensen Corrections
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
        """Advance one step with conservative spike loss and reinjection.

        The old implementation evaluated an exponential sink and a boundary
        condition in separate, non-matching updates.  That preserved
        positivity but lost probability mass.  Here the mass removed by the
        hazard during the reaction substep is measured exactly and used as the
        boundary inflow during the age-transport substep.  The same operation
        is applied to the transported voltage and adaptation moments.
        """
        eps = 1e-14
        courant = self.dt / self.dr
        if not (0.0 < courant <= 1.0):
            raise ValueError(f"upwind CFL requires 0 < dt/dr <= 1; got {courant}")

        self.active_r = min(self.R_max, self.active_r + self.dt)
        # Update the complete finite domain.  A previous moving-front shortcut
        # silently discarded the small numerical tail at the artificial active
        # boundary and was the source of the reported mass drift.
        limit_idx = self.M
        sl = slice(0, limit_idx + 1)

        q = self.q[sl]
        V = self.V_field[sl]
        M = self.M_field[sl]

        # First-order deterministic state update in conservative variables.
        V_det = V + self.dt * (-V / self.tau_m - M + u_t * q)
        M_det = M + self.dt * (-M / self.tau_a + self.kappa * V)

        v_det = np.divide(V_det, q, out=np.zeros_like(V_det), where=q > eps)
        m_det = np.divide(M_det, q, out=np.zeros_like(M_det), where=q > eps)
        rho = self.compute_hazard(v_det, m_det)

        survival = np.exp(-rho * self.dt)
        q_surv = q * survival
        V_surv = V_det * survival
        M_surv = M_det * survival

        # Integrated reaction losses are the conservative reinjection fluxes.
        q_loss = float(np.sum(q - q_surv) * self.dr)
        V_loss = float(np.sum(V_det - V_surv) * self.dr)
        M_loss = float(np.sum(M_det - M_surv) * self.dr)
        A_t = q_loss / self.dt
        V_inflow = V_loss / self.dt
        M_inflow = M_loss / self.dt

        q_new = np.zeros_like(self.q)
        V_new = np.zeros_like(self.V_field)
        M_new = np.zeros_like(self.M_field)

        q_new[0] = q_surv[0] - courant * (q_surv[0] - A_t)
        V_new[0] = V_surv[0] - courant * (V_surv[0] - V_inflow)
        M_new[0] = M_surv[0] - courant * (M_surv[0] - M_inflow)
        if limit_idx >= 1:
            q_new[1:limit_idx + 1] = q_surv[1:] - courant * np.diff(q_surv)
            V_new[1:limit_idx + 1] = V_surv[1:] - courant * np.diff(V_surv)
            M_new[1:limit_idx + 1] = M_surv[1:] - courant * np.diff(M_surv)

        # Only physical density is positivity constrained; moment fields may be signed.
        q_new[:limit_idx + 1] = np.maximum(q_new[:limit_idx + 1], 0.0)
        self.cum_leak += q_surv[-1] * self.dt if limit_idx == self.M else 0.0
        self.q, self.V_field, self.M_field = q_new, V_new, M_new

        self.v = np.divide(
            self.V_field, self.q, out=np.zeros_like(self.q), where=self.q > eps
        )
        self.m = np.divide(
            self.M_field, self.q, out=np.zeros_like(self.q), where=self.q > eps
        )
        self.rho_last_used = np.zeros_like(self.q)
        self.rho_last_used[:limit_idx + 1] = rho
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

