import copy
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from coupling import MeanFieldCoupling
from metrics import ActivityMetrics
from age_joint_gaussian_pde import AgeJointGaussianPDESolver


def test_metric_definition_and_sign():
    mc = np.array([1.0, 2.0, 3.0, 4.0])
    closure = mc + 2.0
    result = ActivityMetrics.compare_activity(mc, closure, dt=1.0, smooth_window=0.0)
    assert np.isclose(result["bias_hz"], 2.0)
    assert np.isclose(result["rmse_hz"], 2.0)
    assert np.isclose(result["centered_nrmse"], 0.0)


def test_nested_configuration_isolation():
    base = {"neuron": {"kappa": 2.0}, "pde": {"use_jensen": True}}
    age_only = copy.deepcopy(base)
    age_only["neuron"]["kappa"] = 0.0
    assert base["neuron"]["kappa"] == 2.0


def test_independent_couplers_do_not_teacher_force():
    mc = MeanFieldCoupling(J=0.1, tau_s=0.01, dt=0.001)
    pde = MeanFieldCoupling(J=0.1, tau_s=0.01, dt=0.001)
    mc.step(10.0)
    pde.step(5.0)
    assert not np.isclose(mc.output, pde.output)


def _joint_params(theta1=2000.0, theta2=-9800.0, sigma=0.003):
    return {
        "neuron": {"tau_m": 0.02, "tau_a": 0.1, "kappa": 2.0, "sigma": sigma},
        "spike_gen": {
            "refractory": 0.005, "theta_0": 2.0,
            "theta_1": theta1, "theta_2": theta2, "type": "exponential",
        },
    }


def test_joint_gaussian_tilt_formula():
    solver = AgeJointGaussianPDESolver(1.0, 1e-4, 0.1, 1e-3, _joint_params())
    v = np.array([2e-4])
    a = np.array([-1e-4])
    vv = np.array([8e-8])
    va = np.array([-2e-8])
    aa = np.array([5e-8])
    rho, tilted_v, tilted_a = solver._hazard_and_tilt(v, a, vv, va, aa)
    expected_shift_v = vv[0] * solver.theta_1 + va[0] * solver.theta_2
    expected_shift_a = va[0] * solver.theta_1 + aa[0] * solver.theta_2
    assert np.isclose(tilted_v[0], v[0] + expected_shift_v)
    assert np.isclose(tilted_a[0], a[0] + expected_shift_a)
    assert rho[0] == 0.0  # first age cell is refractory


def test_joint_gaussian_mass_and_covariance_integrity():
    solver = AgeJointGaussianPDESolver(0.2, 1e-4, 0.5, 1e-3, _joint_params())
    masses = []
    for _ in range(2000):
        activity = solver.step(0.0)
        assert np.isfinite(activity) and activity >= 0.0
        diag = solver.get_diagnostics()
        masses.append(diag["mass"])
        assert diag["min_cov_determinant"] >= -1e-18
    assert abs(masses[-1] - 1.0) < 5e-5
