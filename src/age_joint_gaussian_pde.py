"""Predictive age-structured joint-Gaussian moment closure.

The solver evolves probability, first raw moments, and second raw moments of
voltage and adaptation.  For an exponential escape hazard, Gaussian
exponential tilting supplies the hazard-weighted moments removed by spiking and
reinserted at age zero.  Continuous states are not reset by a spike.
"""

from __future__ import annotations

import numpy as np


class AgeJointGaussianPDESolver:
    """Conservative second-order closure for the joint state ``(V, a)``."""

    def __init__(self, t_len, dt, R_max, dr, params, init_mode="delta0"):
        del t_len  # Kept for API compatibility with the first-order solver.
        self.dt = float(dt)
        self.dr = float(dr)
        self.R_max = float(R_max)
        self.M = int(R_max / dr)
        self.r_grid = np.linspace(0.0, R_max, self.M + 1)
        self.init_mode = init_mode
        self.use_jensen = False

        p = params["neuron"]
        self.tau_m = float(p["tau_m"])
        self.tau_a = float(p["tau_a"])
        self.kappa = float(p["kappa"])
        self.sigma = float(p["sigma"])
        sg = params["spike_gen"]
        self.refractory = float(sg["refractory"])
        self.theta_0 = float(sg["theta_0"])
        self.theta_1 = float(sg["theta_1"])
        self.theta_2 = float(sg["theta_2"])
        if sg.get("type", "exponential") != "exponential":
            raise ValueError("joint-Gaussian closure currently requires an exponential hazard")

        if init_mode != "delta0":
            raise ValueError(f"unsupported init_mode: {init_mode}")
        self.q = np.zeros(self.M + 1)
        spread_time = 0.010
        cells = max(1, int(spread_time / self.dr))
        self.q[:cells] = 1.0
        self.q /= np.sum(self.q) * self.dr

        # Conservative raw-moment fields.
        self.V_field = np.zeros_like(self.q)   # q E[V]
        self.A_field = np.zeros_like(self.q)   # q E[a]
        self.VV_field = np.zeros_like(self.q)  # q E[V^2]
        self.VA_field = np.zeros_like(self.q)  # q E[Va]
        self.AA_field = np.zeros_like(self.q)  # q E[a^2]

        self.cum_leak = 0.0
        self.covariance_projection_total = 0.0
        self.min_cov_eigenvalue = 0.0
        self.rho_last_used = np.zeros_like(self.q)
        self._refresh_intensive_fields(project=True)

    def _conditional_statistics(self, q, V, A, VV, VA, AA, project=False):
        eps = 1e-14
        valid = q > eps
        v = np.divide(V, q, out=np.zeros_like(V), where=valid)
        a = np.divide(A, q, out=np.zeros_like(A), where=valid)
        evv = np.divide(VV, q, out=np.zeros_like(VV), where=valid)
        eva = np.divide(VA, q, out=np.zeros_like(VA), where=valid)
        eaa = np.divide(AA, q, out=np.zeros_like(AA), where=valid)
        with np.errstate(over="ignore", invalid="ignore"):
            var_v = evv - v * v
            cov_va = eva - v * a
            var_a = eaa - a * a
        for name, values in [
            ("mean voltage", v), ("mean adaptation", a),
            ("voltage variance", var_v), ("covariance", cov_va),
            ("adaptation variance", var_a),
        ]:
            if not np.all(np.isfinite(values[valid])):
                raise FloatingPointError(
                    f"joint-Gaussian closure left its finite-moment regime: {name}"
                )

        # Project only roundoff/splitting violations onto the PSD cone.  The
        # accumulated correction is exposed as a diagnostic rather than hidden.
        if project:
            old_v, old_c, old_a = var_v.copy(), cov_va.copy(), var_a.copy()
            var_v = np.maximum(var_v, 0.0)
            var_a = np.maximum(var_a, 0.0)
            bound = np.sqrt(var_v) * np.sqrt(var_a)
            cov_va = np.clip(cov_va, -bound, bound)
            adjustment = (
                np.abs(var_v - old_v) + 2.0 * np.abs(cov_va - old_c)
                + np.abs(var_a - old_a)
            )
            self.covariance_projection_total += float(
                np.sum(adjustment[valid] * q[valid]) * self.dr
            )

        determinant = var_v * var_a - cov_va * cov_va
        trace = var_v + var_a
        discriminant = np.sqrt(np.maximum((var_v - var_a) ** 2 + 4.0 * cov_va ** 2, 0.0))
        minimum_eigenvalue = 0.5 * (trace - discriminant)
        if np.any(valid):
            self.min_cov_eigenvalue = min(
                self.min_cov_eigenvalue, float(np.min(minimum_eigenvalue[valid]))
            )
        return v, a, var_v, cov_va, var_a, determinant

    def _refresh_intensive_fields(self, project=False):
        stats = self._conditional_statistics(
            self.q, self.V_field, self.A_field,
            self.VV_field, self.VA_field, self.AA_field, project=project,
        )
        self.v, self.m, self.var_v, self.cov_va, self.var_a, _ = stats
        if project:
            self.VV_field = self.q * (self.var_v + self.v ** 2)
            self.VA_field = self.q * (self.cov_va + self.v * self.m)
            self.AA_field = self.q * (self.var_a + self.m ** 2)

    def _hazard_and_tilt(self, v, a, var_v, cov_va, var_a):
        shift_v = var_v * self.theta_1 + cov_va * self.theta_2
        shift_a = cov_va * self.theta_1 + var_a * self.theta_2
        exponent = (
            self.theta_0 + self.theta_1 * v + self.theta_2 * a
            + 0.5 * (
                self.theta_1 ** 2 * var_v
                + 2.0 * self.theta_1 * self.theta_2 * cov_va
                + self.theta_2 ** 2 * var_a
            )
        )
        rho = np.exp(np.clip(exponent, -50.0, 50.0))
        rho[self.r_grid[:len(rho)] < self.refractory] = 0.0
        return rho, v + shift_v, a + shift_a

    @staticmethod
    def _advect(field, inflow, courant):
        result = np.zeros_like(field)
        result[0] = field[0] - courant * (field[0] - inflow)
        result[1:] = field[1:] - courant * np.diff(field)
        return result

    def step(self, u_t):
        """Advance deterministic moments, spike selection, and age transport."""
        courant = self.dt / self.dr
        if not (0.0 < courant <= 1.0):
            raise ValueError(f"upwind CFL requires 0 < dt/dr <= 1; got {courant}")

        q = self.q
        V, A = self.V_field, self.A_field
        VV, VA, AA = self.VV_field, self.VA_field, self.AA_field

        # Itô raw-moment dynamics for dV=(-V/tau_m-a+u)dt+sigma dW,
        # da=(-a/tau_a+kappa V)dt.
        V_det = V + self.dt * (-V / self.tau_m - A + u_t * q)
        A_det = A + self.dt * (-A / self.tau_a + self.kappa * V)
        VV_det = VV + self.dt * (
            -2.0 * VV / self.tau_m - 2.0 * VA + 2.0 * u_t * V
            + self.sigma ** 2 * q
        )
        VA_det = VA + self.dt * (
            -(1.0 / self.tau_m + 1.0 / self.tau_a) * VA
            - AA + self.kappa * VV + u_t * A
        )
        AA_det = AA + self.dt * (-2.0 * AA / self.tau_a + 2.0 * self.kappa * VA)

        v, a, var_v, cov_va, var_a, _ = self._conditional_statistics(
            q, V_det, A_det, VV_det, VA_det, AA_det, project=True
        )
        # Use the repaired moments consistently in the reaction substep.
        VV_det = q * (var_v + v * v)
        VA_det = q * (cov_va + v * a)
        AA_det = q * (var_a + a * a)

        rho, tilted_v, tilted_a = self._hazard_and_tilt(v, a, var_v, cov_va, var_a)
        loss_fraction = -np.expm1(-rho * self.dt)
        q_loss_cell = q * loss_fraction

        # Under exponential tilting the covariance is unchanged and the mean
        # shifts by Sigma theta.  These are the hazard-weighted lost moments.
        V_loss_cell = q_loss_cell * tilted_v
        A_loss_cell = q_loss_cell * tilted_a
        VV_loss_cell = q_loss_cell * (var_v + tilted_v ** 2)
        VA_loss_cell = q_loss_cell * (cov_va + tilted_v * tilted_a)
        AA_loss_cell = q_loss_cell * (var_a + tilted_a ** 2)

        q_surv = q - q_loss_cell
        V_surv = V_det - V_loss_cell
        A_surv = A_det - A_loss_cell
        VV_surv = VV_det - VV_loss_cell
        VA_surv = VA_det - VA_loss_cell
        AA_surv = AA_det - AA_loss_cell

        losses = [
            np.sum(x) * self.dr / self.dt for x in
            [q_loss_cell, V_loss_cell, A_loss_cell,
             VV_loss_cell, VA_loss_cell, AA_loss_cell]
        ]
        activity = float(losses[0])
        self.q = self._advect(q_surv, losses[0], courant)
        self.V_field = self._advect(V_surv, losses[1], courant)
        self.A_field = self._advect(A_surv, losses[2], courant)
        self.VV_field = self._advect(VV_surv, losses[3], courant)
        self.VA_field = self._advect(VA_surv, losses[4], courant)
        self.AA_field = self._advect(AA_surv, losses[5], courant)
        self.q = np.maximum(self.q, 0.0)
        self.cum_leak += float(q_surv[-1] * self.dt)
        self.rho_last_used = rho
        self._refresh_intensive_fields(project=True)
        return activity

    def get_diagnostics(self):
        mass = float(np.sum(self.q) * self.dr)
        k = max(1, int(0.5 / self.dr))
        tail_mass = float(np.sum(self.q[-k:]) * self.dr)
        determinant = self.var_v * self.var_a - self.cov_va ** 2
        valid = self.q > 1e-12
        return {
            "mass": mass,
            "tail_mass": tail_mass,
            "cum_leak": self.cum_leak,
            "covariance_projection_total": self.covariance_projection_total,
            "min_cov_eigenvalue": self.min_cov_eigenvalue,
            "min_cov_determinant": float(np.min(determinant[valid])) if np.any(valid) else 0.0,
        }
