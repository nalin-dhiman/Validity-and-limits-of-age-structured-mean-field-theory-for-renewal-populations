"""Microscopic population simulation with an optional closed-loop PDE closure."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from age_adaptation_pde import AgeAdaptationPDESolver
from coupling import MeanFieldCoupling
from numerics_numba import seed_numba, step_neuron_numba
from simulation_logger import SimulationLogger
from stimuli import StimulusGenerator
from utils import load_config, set_seed


class VectorizedNeuron:
    def __init__(self, params, N, seed=None):
        self.N = int(N)
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
        self.h_type_code = 1 if sg.get("type", "exponential") == "softplus" else 0
        self.V = np.zeros(self.N)
        self.a = np.zeros(self.N)
        self.last_spike_time = np.full(self.N, -1000.0)
        if seed is not None:
            np.random.seed(int(seed))
            seed_numba(int(seed))

    def step(self, t, dt, u_t, shared_noise=0.0, c_sqrt_shared=0.0, c_sqrt_indep=1.0):
        return step_neuron_numba(
            self.V, self.a, self.last_spike_time, t, dt, u_t,
            shared_noise, self.N, self.tau_m, self.tau_a, self.kappa,
            self.sigma, self.refractory, self.theta_0, self.theta_1,
            self.theta_2, self.h_type_code, c_sqrt_shared, c_sqrt_indep,
        )


def run_population_sim(config, output_file, online_pde=False):
    logger = SimulationLogger("simulate_population.py", config, output_file)
    try:
        _run_population_sim_inner(config, output_file, online_pde)
    except Exception as exc:
        print(f"CRITICAL FAILURE: {exc}")
        traceback.print_exc()
        logger.close(exit_code=1)
        raise
    else:
        logger.close(exit_code=0)


def _run_population_sim_inner(config, output_file, online_pde=False):
    config = copy.deepcopy(config)
    dt = float(config["defaults"]["dt"])
    duration = float(config.get("duration", 2.0))
    seed = int(config["defaults"]["seed"])
    N = int(config["population"]["N"])
    c = float(config["population"].get("shared_noise_fraction", 0.0))
    J = float(config["coupling"]["J"])
    tau_s = float(config["coupling"]["tau_s"])
    set_seed(seed)

    stimulus = copy.deepcopy(config["stimulus"])
    stimulus.pop("type", None)
    stimulus_seed = seed + 100_000
    u_ext = StimulusGenerator(dt).generate_ou(
        duration, seed=stimulus_seed, **stimulus
    )
    n_steps = u_ext.size

    pop = VectorizedNeuron(config, N, seed=seed)
    pop.last_spike_time = -np.random.uniform(0.0, 0.010, N)
    mc_coupler = MeanFieldCoupling(J, tau_s, dt)

    pde = None
    pde_coupler = None
    if online_pde:
        pde_config = copy.deepcopy(config)
        kappa_override = pde_config["pde"].get("model_kappa_override")
        if kappa_override is not None:
            pde_config["neuron"]["kappa"] = float(kappa_override)
        R_max = pde_config["pde"].get("R_max", "auto")
        if R_max == "auto":
            R_max = float(pde_config["pde"].get("age_domain_s", 2.0))
        use_jensen = bool(pde_config["pde"].get("use_jensen", True))
        pde = AgeAdaptationPDESolver(
            duration, dt, float(R_max), float(pde_config["pde"]["dr"]),
            pde_config, use_jensen=use_jensen,
        )
        pde_coupler = MeanFieldCoupling(J, tau_s, dt)

    A_mc = np.zeros(n_steps)
    A_pde = np.zeros(n_steps) if online_pde else None
    mass = np.zeros(n_steps) if online_pde else None
    tail = np.zeros(n_steps) if online_pde else None
    mc_sqrt = np.sqrt(c)
    indep_sqrt = np.sqrt(1.0 - c)
    started = time.perf_counter()

    for i in range(n_steps):
        t = i * dt
        u_mc = u_ext[i] + mc_coupler.output
        spikes, _ = pop.step(
            t, dt, u_mc, shared_noise=np.random.normal(),
            c_sqrt_shared=mc_sqrt, c_sqrt_indep=indep_sqrt,
        )
        A_mc[i] = np.sum(spikes) / (N * dt)
        mc_coupler.step(A_mc[i])

        if online_pde:
            # The closure evolves its own feedback state.  No MC teacher forcing.
            u_pde = u_ext[i] + pde_coupler.output
            A_pde[i] = pde.step(u_pde)
            pde_coupler.step(A_pde[i])
            diag = pde.get_diagnostics()
            mass[i] = diag["mass"]
            tail[i] = diag["tail_mass"]

    runtime_s = time.perf_counter() - started
    provenance = {
        "seed": seed,
        "stimulus_seed": stimulus_seed,
        "mc_kappa": float(config["neuron"]["kappa"]),
        "J": J,
        "metric_normalization": "smoothed_mc_temporal_std",
        "coupling_mode": "independent_closed_loop" if online_pde else "mc_only",
        "runtime_s": runtime_s,
    }
    save = {
        "A": A_mc, "u_ext": u_ext, "config": config,
        "provenance": provenance, "runtime_s": runtime_s,
        "pop_V": pop.V, "pop_a": pop.a,
        "pop_last_spike_time": pop.last_spike_time,
    }
    if online_pde:
        pde_metrics = {
            "solver": "AgeAdaptationPDESolver",
            "mass_min": float(np.min(mass)),
            "mass_final": float(mass[-1]),
            "mass_error_max": float(np.max(np.abs(mass - 1.0))),
            "tail_mass_max": float(np.max(tail)),
            "cum_leak_final": float(pde.cum_leak),
            "R_max": float(pde.R_max),
            "dr": float(pde.dr),
            "dt": dt,
        }
        provenance.update({
            "pde_kappa": float(pde.kappa),
            "use_jensen": bool(pde.use_jensen),
            "solver_identity": "AgeAdaptationPDESolver",
        })
        save.update({
            "A_pde": A_pde, "A_pde_adapt": A_pde,
            "pde_mass": mass, "pde_tail": tail,
            "pde_metrics": pde_metrics,
        })
    np.savez(output_file, **save)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--outfile", default="results/pop_sim.npz")
    parser.add_argument("--online-pde", action="store_true")
    args = parser.parse_args()
    run_population_sim(load_config(args.config), args.outfile, args.online_pde)
