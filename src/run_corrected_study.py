"""Generate the audited numerical evidence for the corrective PRE revision.

All closures are compared against the same adapted microscopic model and the
same external-input realization.  Coupled MC and PDE systems evolve independent
feedback states.  Every CSV records the relevant configuration provenance.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from age_adaptation_pde import AgeAdaptationPDESolver
from age_joint_gaussian_pde import AgeJointGaussianPDESolver
from coupling import MeanFieldCoupling
from metrics import ActivityMetrics
from simulate_population import VectorizedNeuron
from stimuli import StimulusGenerator
from utils import ensure_dir, load_config, set_seed


VARIANTS = {
    "adaptation_neglecting": {"kappa": 0.0, "jensen": False},
    "mean_only": {"kappa": None, "jensen": False},
    "variance_corrected": {"kappa": None, "jensen": True},
    "joint_gaussian": {"kappa": None, "jensen": False},
}


def external_input(cfg, duration, seed):
    params = copy.deepcopy(cfg["stimulus"])
    params.pop("type", None)
    return StimulusGenerator(cfg["defaults"]["dt"]).generate_ou(
        duration, seed=int(seed) + 100_000, **params
    )


def simulate_mc(cfg, u_ext, N, seed, J=0.0):
    cfg = copy.deepcopy(cfg)
    dt = float(cfg["defaults"]["dt"])
    cfg["population"]["N"] = int(N)
    cfg["defaults"]["seed"] = int(seed)
    cfg["coupling"]["J"] = float(J)
    set_seed(seed)
    pop = VectorizedNeuron(cfg, N, seed=seed)
    pop.last_spike_time = -np.random.uniform(0.0, 0.010, int(N))
    coupler = MeanFieldCoupling(J, cfg["coupling"]["tau_s"], dt)
    c = float(cfg["population"].get("shared_noise_fraction", 0.0))
    out = np.zeros(len(u_ext))
    started = time.perf_counter()
    for i, drive in enumerate(u_ext):
        spikes, _ = pop.step(
            i * dt, dt, drive + coupler.output,
            shared_noise=np.random.normal(),
            c_sqrt_shared=np.sqrt(c), c_sqrt_indep=np.sqrt(1.0 - c),
        )
        out[i] = np.sum(spikes) / (N * dt)
        coupler.step(out[i])
    return out, time.perf_counter() - started


def simulate_pde(cfg, u_ext, variant, J=0.0, dr=None):
    cfg = copy.deepcopy(cfg)
    spec = VARIANTS[variant]
    if spec["kappa"] is not None:
        cfg["neuron"]["kappa"] = float(spec["kappa"])
    dt = float(cfg["defaults"]["dt"])
    dr = float(dr if dr is not None else cfg["pde"]["dr"])
    R_max = cfg["pde"].get("R_max", "auto")
    if R_max == "auto":
        R_max = float(cfg["pde"].get("age_domain_s", 2.0))
    if variant == "joint_gaussian":
        solver = AgeJointGaussianPDESolver(
            len(u_ext) * dt, dt, float(R_max), dr, cfg,
        )
    else:
        solver = AgeAdaptationPDESolver(
            len(u_ext) * dt, dt, float(R_max), dr, cfg,
            use_jensen=bool(spec["jensen"]),
        )
    coupler = MeanFieldCoupling(float(J), cfg["coupling"]["tau_s"], dt)
    activity = np.zeros(len(u_ext))
    mass = np.zeros(len(u_ext))
    tail = np.zeros(len(u_ext))
    min_cov_eigenvalue = 0.0
    covariance_projection_total = 0.0
    started = time.perf_counter()
    for i, drive in enumerate(u_ext):
        activity[i] = solver.step(drive + coupler.output)
        coupler.step(activity[i])
        diag = solver.get_diagnostics()
        mass[i] = diag["mass"]
        tail[i] = diag["tail_mass"]
        min_cov_eigenvalue = min(min_cov_eigenvalue, float(diag.get("min_cov_eigenvalue", 0.0)))
        covariance_projection_total = float(diag.get("covariance_projection_total", 0.0))
    elapsed = time.perf_counter() - started
    diagnostics = {
        "mass_min": float(mass.min()),
        "mass_final": float(mass[-1]),
        "mass_error_max": float(np.max(np.abs(mass - 1.0))),
        "tail_mass_max": float(tail.max()),
        "cum_leak_final": float(solver.cum_leak),
        "runtime_s": elapsed,
        "pde_kappa": float(solver.kappa),
        "use_jensen": bool(solver.use_jensen),
        "dr": dr,
        "dt": dt,
        "R_max": float(R_max),
        "dynamic_joint_moments": variant == "joint_gaussian",
        "min_cov_eigenvalue": min_cov_eigenvalue,
        "covariance_projection_total": covariance_projection_total,
    }
    return activity, diagnostics


def metric_row(mc, closure, dt, burn_s, windows=(0.02, 0.05)):
    burn = int(burn_s / dt)
    row = {}
    for window in windows:
        m = ActivityMetrics.compare_activity(
            mc[burn:], closure[burn:], dt, smooth_window=window
        )
        suffix = f"_{int(window * 1000)}ms"
        for key, value in m.items():
            row[f"{key}{suffix}"] = value
    return row


def run_bias(cfg, outdir, duration=20.0):
    rows = []
    for seed in range(5):
        u = external_input(cfg, duration, seed)
        mc, mc_runtime = simulate_mc(cfg, u, N=5000, seed=seed)
        for variant in VARIANTS:
            closure, diag = simulate_pde(cfg, u, variant)
            row = {"Closure": variant, "Seed": seed, "N": 5000,
                   "Duration_s": duration, "MC_runtime_s": mc_runtime, **diag}
            row.update(metric_row(mc, closure, cfg["defaults"]["dt"], burn_s=5.0))
            rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "bias_corrected.csv", index=False)


def fit_offset(N, y):
    def model(n, a, b):
        return a / np.sqrt(n) + b
    popt, pcov = curve_fit(model, N, y, p0=(20.0, max(0.0, y[-1])))
    err = np.sqrt(np.diag(pcov))
    slope, _ = np.polyfit(np.log(N), np.log(y), 1)
    return popt, err, -float(slope)


def write_scaling_fits(df, outdir):
    """Write fits using dimensional total RMSE and centered RMSE."""
    fits = []
    for variant in [v for v in VARIANTS if v in set(df.Closure)]:
        grouped = df[df.Closure == variant].groupby("N")
        total = grouped["rmse_hz_50ms"].mean()
        centered = grouped["crmse_hz_50ms"].mean()
        popt, err, total_alpha = fit_offset(
            total.index.values.astype(float), total.values
        )
        centered_alpha = -float(np.polyfit(
            np.log(centered.index.values.astype(float)), np.log(centered.values), 1
        )[0])
        fits.append({
            "Closure": variant, "total_a": popt[0], "total_bias_floor_hz": popt[1],
            "total_a_se": err[0], "bias_floor_se_hz": err[1],
            "total_power_alpha": total_alpha,
            "centered_power_alpha": centered_alpha,
        })
    pd.DataFrame(fits).to_csv(outdir / "scaling_fit_corrected.csv", index=False)


def run_scaling(cfg, outdir, duration=15.0):
    Ns = [1000, 2000, 4000, 8000, 16000]
    rows = []
    for seed in [0, 1, 2]:
        u = external_input(cfg, duration, seed)
        closures = {v: simulate_pde(cfg, u, v)[0] for v in VARIANTS}
        for N in Ns:
            mc, runtime = simulate_mc(cfg, u, N=N, seed=seed)
            for variant, closure in closures.items():
                row = {"N": N, "Closure": variant, "Seed": seed,
                       "Duration_s": duration, "MC_runtime_s": runtime}
                row.update(metric_row(mc, closure, cfg["defaults"]["dt"], burn_s=3.0))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "scaling_corrected.csv", index=False)
    write_scaling_fits(df, outdir)


def run_scaling_refit(cfg, outdir):
    del cfg
    write_scaling_fits(pd.read_csv(outdir / "scaling_corrected.csv"), outdir)


def run_coupling(cfg, outdir, duration=10.0):
    rows = []
    A_ref_hz = 8.03
    V_ref = 1e-3
    tau_m = float(cfg["neuron"]["tau_m"])
    for J in [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]:
        for seed in [7, 11, 13]:
            u = external_input(cfg, duration, seed)
            mc, mc_runtime = simulate_mc(cfg, u, N=5000, seed=seed, J=J)
            for variant in ["variance_corrected", "joint_gaussian"]:
                closure, diag = simulate_pde(cfg, u, variant, J=J)
                row = {"J": J, "g": tau_m * J * A_ref_hz / V_ref,
                       "Closure": variant, "Seed": seed, "N": 5000,
                       "Duration_s": duration, "MC_runtime_s": mc_runtime, **diag}
                row.update(metric_row(mc, closure, cfg["defaults"]["dt"], burn_s=2.0,
                                      windows=(0.05,)))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "coupling_closed_loop.csv", index=False)
    summary = df.groupby(["Closure", "J", "g"]).agg(
        nrmse_mean=("nrmse_50ms", "mean"),
        nrmse_std=("nrmse_50ms", "std"),
        bias_mean_hz=("bias_hz_50ms", "mean"),
        mass_error_max=("mass_error_max", "max"),
        tail_mass_max=("tail_mass_max", "max"),
    ).reset_index()
    summary.to_csv(outdir / "coupling_closed_loop_summary.csv", index=False)


def run_jensen_sweep(cfg, outdir, duration=12.0):
    rows = []
    baseline_path = outdir / "bias_corrected.csv"
    for sigma in [0.0015, 0.0030, 0.0045]:
        local = copy.deepcopy(cfg)
        local["neuron"]["sigma"] = sigma
        chi_inf = (local["spike_gen"]["theta_1"] ** 2 * sigma ** 2
                   * local["neuron"]["tau_m"] / 4.0)
        for seed in [0, 1, 2]:
            u = external_input(local, duration, seed + 30)
            mc, _ = simulate_mc(local, u, N=5000, seed=seed + 30)
            for variant in ["mean_only", "variance_corrected"]:
                closure, diag = simulate_pde(local, u, variant)
                row = {"sigma": sigma, "chi_inf": chi_inf, "Seed": seed,
                       "Closure": variant, **diag}
                row.update(metric_row(mc, closure, local["defaults"]["dt"],
                                      burn_s=2.0, windows=(0.05,)))
                rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "jensen_sweep.csv", index=False)


def run_convergence(cfg, outdir, duration=5.0):
    rows = []
    u = external_input(cfg, duration, 91)
    finest = None
    for dr in [0.002, 0.001, 0.0005]:
        trace, diag = simulate_pde(cfg, u, "variance_corrected", J=1e-3, dr=dr)
        if dr == 0.0005:
            finest = trace
        rows.append({"dr": dr, **diag, "trace": trace})
    output = []
    for row in rows:
        trace = row.pop("trace")
        comparison = ActivityMetrics.compare_activity(
            finest[int(1.0/cfg["defaults"]["dt"]):],
            trace[int(1.0/cfg["defaults"]["dt"]):],
            cfg["defaults"]["dt"], smooth_window=0.05,
        )
        output.append({**row, "rmse_vs_finest_hz": comparison["rmse_hz"]})
    pd.DataFrame(output).to_csv(outdir / "grid_convergence.csv", index=False)


def run_performance(cfg, outdir, duration=1.0):
    # Compile once so JIT overhead is excluded from reported timings.
    warm_u = external_input(cfg, 0.02, 501)
    simulate_mc(cfg, warm_u, N=100, seed=501)
    simulate_pde(cfg, warm_u, "variance_corrected")
    u = external_input(cfg, duration, 502)
    pde, diag = simulate_pde(cfg, u, "variance_corrected")
    rows = []
    for N in [1000, 5000, 20000, 50000]:
        mc, runtime = simulate_mc(cfg, u, N=N, seed=502)
        metrics = metric_row(mc, pde, cfg["defaults"]["dt"], burn_s=0.2,
                             windows=(0.05,))
        rows.append({"N": N, "MC_runtime_s": runtime,
                     "PDE_runtime_s": diag["runtime_s"],
                     "speedup_MC_over_PDE": runtime / diag["runtime_s"], **metrics})
    pd.DataFrame(rows).to_csv(outdir / "performance.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--outdir", default="results/corrected_revision")
    parser.add_argument("--section", choices=["all", "bias", "scaling", "coupling",
                                               "scaling_refit", "jensen", "convergence",
                                               "performance"],
                        default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    with open(outdir / "study_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    jobs = {
        "bias": run_bias, "scaling": run_scaling, "coupling": run_coupling,
        "scaling_refit": run_scaling_refit,
        "jensen": run_jensen_sweep, "convergence": run_convergence,
        "performance": run_performance,
    }
    selected = jobs if args.section == "all" else {args.section: jobs[args.section]}
    for name, function in selected.items():
        print(f"[corrected study] starting {name}", flush=True)
        function(cfg, outdir)
        print(f"[corrected study] completed {name}", flush=True)


if __name__ == "__main__":
    main()
