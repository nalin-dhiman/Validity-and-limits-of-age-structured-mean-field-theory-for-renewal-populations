"""Run predictive joint-closure validation and accuracy-cost studies."""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from metrics import ActivityMetrics
from run_corrected_study import (
    external_input,
    metric_row,
    simulate_mc,
    simulate_pde,
)
from simulate_population import VectorizedNeuron
from coupling import MeanFieldCoupling
from age_joint_gaussian_pde import AgeJointGaussianPDESolver
from utils import ensure_dir, load_config, set_seed


def run_joint_cumulants(cfg, outdir, duration=6.0, N=30000):
    """Compare MC conditional moments with the predictive joint closure."""
    local = copy.deepcopy(cfg)
    local["coupling"]["J"] = 0.0
    seed = 610
    dt = float(local["defaults"]["dt"])
    u = external_input(local, duration, seed)
    set_seed(seed)
    pop = VectorizedNeuron(local, N, seed=seed)
    pop.last_spike_time = -np.random.uniform(0.0, 0.010, N)
    coupler = MeanFieldCoupling(0.0, local["coupling"]["tau_s"], dt)
    solver = AgeJointGaussianPDESolver(
        duration, dt, float(local["pde"].get("age_domain_s", 2.0)),
        float(local["pde"]["dr"]), local,
    )

    bin_width = 0.005
    max_age = 0.5
    n_bins = int(max_age / bin_width)
    accum = {name: np.zeros(n_bins) for name in
             ["count", "V", "a", "V2", "a2", "Va", "hazard"]}
    pde_accum = {name: np.zeros(n_bins) for name in
                 ["weight", "V", "a", "V2", "a2", "Va", "hazard"]}
    sample_stride = max(1, int(0.002 / dt))
    burn_step = int(1.0 / dt)

    for i, drive in enumerate(u):
        t = i * dt
        spikes, hazard = pop.step(
            t, dt, drive + coupler.output, shared_noise=np.random.normal(),
            c_sqrt_shared=0.0, c_sqrt_indep=1.0,
        )
        activity = np.sum(spikes) / (N * dt)
        coupler.step(activity)
        solver.step(drive)
        if i < burn_step or i % sample_stride:
            continue
        ages = (i + 1) * dt - pop.last_spike_time
        index = (ages / bin_width).astype(np.int64)
        valid = (index >= 0) & (index < n_bins)
        idx = index[valid]
        V = pop.V[valid]
        a = pop.a[valid]
        h = hazard[valid]
        accum["count"] += np.bincount(idx, minlength=n_bins)
        accum["V"] += np.bincount(idx, weights=V, minlength=n_bins)
        accum["a"] += np.bincount(idx, weights=a, minlength=n_bins)
        accum["V2"] += np.bincount(idx, weights=V * V, minlength=n_bins)
        accum["a2"] += np.bincount(idx, weights=a * a, minlength=n_bins)
        accum["Va"] += np.bincount(idx, weights=V * a, minlength=n_bins)
        accum["hazard"] += np.bincount(idx, weights=h, minlength=n_bins)

        target_age = (np.arange(n_bins) + 0.5) * bin_width
        q_age = np.interp(target_age, solver.r_grid, solver.q)
        v_age = np.interp(target_age, solver.r_grid, solver.v)
        a_age = np.interp(target_age, solver.r_grid, solver.m)
        vv_age = np.interp(target_age, solver.r_grid, solver.var_v + solver.v ** 2)
        aa_age = np.interp(target_age, solver.r_grid, solver.var_a + solver.m ** 2)
        va_age = np.interp(target_age, solver.r_grid, solver.cov_va + solver.v * solver.m)
        h_age = np.interp(target_age, solver.r_grid, solver.rho_last_used)
        pde_accum["weight"] += q_age
        pde_accum["V"] += q_age * v_age
        pde_accum["a"] += q_age * a_age
        pde_accum["V2"] += q_age * vv_age
        pde_accum["a2"] += q_age * aa_age
        pde_accum["Va"] += q_age * va_age
        pde_accum["hazard"] += q_age * h_age

    count = accum["count"]
    safe = np.maximum(count, 1.0)
    mean_v = accum["V"] / safe
    mean_a = accum["a"] / safe
    var_v = np.maximum(accum["V2"] / safe - mean_v ** 2, 0.0)
    var_a = np.maximum(accum["a2"] / safe - mean_a ** 2, 0.0)
    cov_va = accum["Va"] / safe - mean_v * mean_a
    empirical = accum["hazard"] / safe
    theta0 = float(local["spike_gen"]["theta_0"])
    theta1 = float(local["spike_gen"]["theta_1"])
    theta2 = float(local["spike_gen"]["theta_2"])
    base = theta0 + theta1 * mean_v + theta2 * mean_a
    term_v = 0.5 * theta1 ** 2 * var_v
    term_a = 0.5 * theta2 ** 2 * var_a
    term_cov = theta1 * theta2 * cov_va
    mean_hazard = np.exp(np.clip(base, -50, 50))
    voltage_only = np.exp(np.clip(base + term_v, -50, 50))
    full_gaussian = np.exp(np.clip(base + term_v + term_a + term_cov, -50, 50))
    age = (np.arange(n_bins) + 0.5) * bin_width
    refractory = age < float(local["spike_gen"]["refractory"])
    for values in [mean_hazard, voltage_only, full_gaussian]:
        values[refractory] = 0.0

    pde_weight = np.maximum(pde_accum["weight"], 1e-30)
    pde_v = pde_accum["V"] / pde_weight
    pde_a = pde_accum["a"] / pde_weight
    pde_var_v = np.maximum(pde_accum["V2"] / pde_weight - pde_v ** 2, 0.0)
    pde_var_a = np.maximum(pde_accum["a2"] / pde_weight - pde_a ** 2, 0.0)
    pde_cov = pde_accum["Va"] / pde_weight - pde_v * pde_a
    pde_hazard = pde_accum["hazard"] / pde_weight

    df = pd.DataFrame({
        "age_s": age, "count": count, "mean_v": mean_v, "mean_a": mean_a,
        "var_v": var_v, "var_a": var_a, "cov_va": cov_va,
        "term_v": term_v, "term_a": term_a, "term_cov": term_cov,
        "hazard_empirical_hz": empirical, "hazard_mean_hz": mean_hazard,
        "hazard_voltage_only_hz": voltage_only,
        "hazard_full_gaussian_hz": full_gaussian,
        "pde_mean_v": pde_v, "pde_mean_a": pde_a,
        "pde_var_v": pde_var_v, "pde_var_a": pde_var_a,
        "pde_cov_va": pde_cov, "pde_hazard_hz": pde_hazard,
    })
    df.to_csv(outdir / "joint_cumulant_diagnostic.csv", index=False)


def run_joint_baseline(cfg, outdir, duration=10.0, N=5000):
    """Five-seed baseline hierarchy including the predictive joint closure."""
    rows = []
    for seed in range(5):
        u = external_input(cfg, duration, 1000 + seed)
        mc, mc_runtime = simulate_mc(cfg, u, N=N, seed=1000 + seed)
        for variant in ["mean_only", "variance_corrected", "joint_gaussian"]:
            trace, diag = simulate_pde(cfg, u, variant)
            metrics = metric_row(mc, trace, cfg["defaults"]["dt"], burn_s=2.0,
                                 windows=(0.02, 0.05, 0.10))
            rows.append({
                "Closure": variant, "Seed": seed, "N": N,
                "Duration_s": duration, "MC_runtime_s": mc_runtime,
                **metrics, **diag,
            })
    pd.DataFrame(rows).to_csv(outdir / "joint_closure_baseline.csv", index=False)


def _cross_chi_case(args):
    cfg, duration, N, chi, theta1, seed_offset, seed = args
    local = copy.deepcopy(cfg)
    tau_m = float(local["neuron"]["tau_m"])
    sigma = np.sqrt(4.0 * chi / (theta1 ** 2 * tau_m))
    local["spike_gen"]["theta_1"] = theta1
    local["neuron"]["sigma"] = float(sigma)
    u = external_input(local, duration, seed)
    mc, _ = simulate_mc(local, u, N=N, seed=seed)
    rows = []
    for variant in ["mean_only", "variance_corrected", "joint_gaussian"]:
        closure, diag = simulate_pde(local, u, variant)
        metrics = metric_row(mc, closure, local["defaults"]["dt"], burn_s=2.0,
                             windows=(0.05,))
        rows.append({
            "chi_target": chi, "theta1": theta1, "sigma": sigma,
            "Seed": seed_offset, "Closure": variant,
            "mc_mean_hz": metrics["mc_mean_hz_50ms"], **metrics, **diag,
        })
    return rows


def run_cross_chi(cfg, outdir, duration=8.0, N=5000):
    """Vary theta1 and sigma independently at matched convexity loads."""
    targets = [0.045, 0.180, 0.405]
    theta_values = [1500.0, 2000.0, 2500.0]
    tasks = [
        (cfg, duration, N, chi, theta1, seed_offset, seed)
        for chi in targets for theta1 in theta_values
        for seed_offset, seed in enumerate([710, 711, 712, 713, 714])
    ]
    workers = min(6, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        nested = list(pool.map(_cross_chi_case, tasks))
    rows = [row for group in nested for row in group]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cross_parameter_chi.csv", index=False)
    pivot = df.pivot_table(
        index=["chi_target", "theta1", "sigma", "Seed"], columns="Closure",
        values="rmse_hz_50ms",
    ).reset_index()
    pivot["rmse_improvement_factor"] = pivot["mean_only"] / pivot["variance_corrected"]
    pivot["joint_improvement_over_mean"] = pivot["mean_only"] / pivot["joint_gaussian"]
    pivot["joint_improvement_over_fixed"] = pivot["variance_corrected"] / pivot["joint_gaussian"]
    pivot.to_csv(outdir / "cross_parameter_chi_improvement.csv", index=False)


def _adaptation_case(args):
    cfg, duration, N, kappa, replicate, seed = args
    local = copy.deepcopy(cfg)
    local["neuron"]["kappa"] = kappa
    u = external_input(local, duration, seed)
    mc, runtime = simulate_mc(local, u, N=N, seed=seed)
    rows = []
    for variant in ["mean_only", "variance_corrected", "joint_gaussian"]:
        trace, diag = simulate_pde(local, u, variant)
        metrics = metric_row(mc, trace, local["defaults"]["dt"], burn_s=2.0,
                             windows=(0.05,))
        rows.append({
            "kappa": kappa, "Seed": replicate, "Closure": variant,
            "N": N, "Duration_s": duration, "MC_runtime_s": runtime,
            **metrics, **diag,
        })
    return rows


def run_adaptation_sweep(cfg, outdir, duration=8.0, N=5000):
    """Test closure hierarchy as adaptation coupling changes at fixed chi."""
    tasks = [
        (cfg, duration, N, kappa, replicate, seed)
        for kappa in [0.5, 1.0, 2.0, 4.0]
        for replicate, seed in enumerate([1010, 1011, 1012, 1013, 1014])
    ]
    workers = min(6, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        nested = list(pool.map(_adaptation_case, tasks))
    rows = [row for group in nested for row in group]
    pd.DataFrame(rows).to_csv(outdir / "adaptation_strength_sweep.csv", index=False)


def run_joint_grid(cfg, outdir, duration=5.0):
    """Check joint-closure age-grid convergence and covariance integrity."""
    u = external_input(cfg, duration, 1090)
    traces, rows = {}, []
    for dr in [0.002, 0.001, 0.0005]:
        trace, diag = simulate_pde(cfg, u, "joint_gaussian", dr=dr)
        traces[dr] = trace
        rows.append({"dr": dr, **diag})
    dt = float(cfg["defaults"]["dt"])
    burn = int(1.0 / dt)
    finest = traces[0.0005]
    for row in rows:
        metrics = ActivityMetrics.compare_activity(
            finest[burn:], traces[row["dr"]][burn:], dt, smooth_window=0.05
        )
        row.update({f"vs_finest_{key}": value for key, value in metrics.items()})
    pd.DataFrame(rows).to_csv(outdir / "joint_grid_convergence.csv", index=False)


def run_joint_timestep(cfg, outdir, duration=2.0):
    """Check the first-order hazard-selection splitting by halving dt."""
    traces, rows = {}, []
    for dt in [0.0001, 0.00005]:
        local = copy.deepcopy(cfg)
        local["defaults"]["dt"] = dt
        time_s = np.arange(int(duration / dt)) * dt
        u = np.zeros_like(time_s)
        u[time_s >= 0.5] = 0.003
        trace, diag = simulate_pde(local, u, "joint_gaussian", dr=0.001)
        traces[dt] = trace
        rows.append({"dt": dt, **diag})
    fine_on_coarse = traces[0.00005][::2][:len(traces[0.0001])]
    burn = int(0.2 / 0.0001)
    metrics = ActivityMetrics.compare_activity(
        fine_on_coarse[burn:], traces[0.0001][burn:], 0.0001, smooth_window=0.02
    )
    rows[0].update({f"vs_half_dt_{key}": value for key, value in metrics.items()})
    pd.DataFrame(rows).to_csv(outdir / "joint_timestep_convergence.csv", index=False)


def _response_features(time_s, trace, onset=2.0):
    baseline_mask = (time_s >= 1.0) & (time_s < 1.8)
    response_mask = (time_s >= onset) & (time_s < onset + 1.0)
    late_mask = (time_s >= 5.0) & (time_s < 6.0)
    baseline = float(np.mean(trace[baseline_mask]))
    segment = trace[response_mask]
    segment_t = time_s[response_mask]
    peak_idx = int(np.argmax(segment))
    return {
        "baseline_hz": baseline,
        "peak_hz": float(segment[peak_idx]),
        "peak_latency_s": float(segment_t[peak_idx] - onset),
        "late_rate_hz": float(np.mean(trace[late_mask])),
        "integrated_excess_spikes": float(np.trapz(trace[time_s >= onset] - baseline,
                                                    time_s[time_s >= onset])),
    }


def run_transient(cfg, outdir, duration=6.0, N=20000):
    """Compare three closures over weak, baseline, and strong input steps."""
    dt = float(cfg["defaults"]["dt"])
    time_s = np.arange(int(duration / dt)) * dt
    onset = 2.0
    trace_frames, features = [], []
    for amplitude in [0.0015, 0.003, 0.006]:
        u = np.zeros_like(time_s)
        u[time_s >= onset] = amplitude
        mc_traces = []
        for seed in [810, 811, 812]:
            # Common random seeds across amplitudes reduce comparison noise.
            mc, _ = simulate_mc(cfg, u, N=N, seed=seed)
            mc_traces.append(mc)
        smooth = {"MC": gaussian_filter1d(np.mean(mc_traces, axis=0), 0.02 / dt)}
        for variant in ["mean_only", "variance_corrected", "joint_gaussian"]:
            trace, _ = simulate_pde(cfg, u, variant)
            smooth[variant] = gaussian_filter1d(trace, 0.02 / dt)
        stride = max(1, int(0.001 / dt))
        trace_frames.append(pd.DataFrame({
            "amplitude": amplitude, "time_s": time_s[::stride],
            "input_u": u[::stride], "mc_hz": smooth["MC"][::stride],
            "mean_only_hz": smooth["mean_only"][::stride],
            "variance_corrected_hz": smooth["variance_corrected"][::stride],
            "joint_gaussian_hz": smooth["joint_gaussian"][::stride],
        }))
        for model, trace in smooth.items():
            row = {"amplitude": amplitude, "Model": model,
                   **_response_features(time_s, trace, onset=onset)}
            if model != "MC":
                row.update(ActivityMetrics.compare_activity(
                    smooth["MC"][int(1.0/dt):], trace[int(1.0/dt):], dt,
                    smooth_window=0.0,
                ))
            features.append(row)
    pd.concat(trace_frames, ignore_index=True).to_csv(
        outdir / "transient_step_traces.csv", index=False
    )
    pd.DataFrame(features).to_csv(outdir / "transient_step_features.csv", index=False)


def run_accuracy_cost(cfg, outdir, duration=2.0):
    """Measure error and cost against an averaged large-N microscopic reference."""
    dt = float(cfg["defaults"]["dt"])
    warm = np.zeros(int(0.02 / dt))
    simulate_mc(cfg, warm, N=100, seed=900)
    simulate_pde(cfg, warm, "variance_corrected", dr=0.001)
    u = external_input(cfg, duration, 901)

    reference_traces = []
    reference_runtime = []
    for seed in [901, 902]:
        trace, runtime = simulate_mc(cfg, u, N=100000, seed=seed)
        reference_traces.append(trace)
        reference_runtime.append(runtime)
    reference = np.mean(reference_traces, axis=0)
    rows = []
    for N in [1000, 5000, 20000, 50000]:
        for replicate, seed in enumerate([911, 912, 913]):
            trace, runtime = simulate_mc(cfg, u, N=N, seed=seed)
            metrics = ActivityMetrics.compare_activity(
                reference[int(0.3/dt):], trace[int(0.3/dt):], dt, 0.05
            )
            rows.append({
                "Method": "MC", "resolution": N, "replicate": replicate,
                "runtime_s": runtime, "state_memory_mb": 24.0 * N / 1e6,
                **metrics,
            })
    R_max = float(cfg["pde"].get("age_domain_s", 2.0))
    for closure, arrays_per_cell in [("variance_corrected", 3), ("joint_gaussian", 6)]:
        for dr in [0.002, 0.001, 0.0005]:
            trace, diag = simulate_pde(cfg, u, closure, dr=dr)
            metrics = ActivityMetrics.compare_activity(
                reference[int(0.3/dt):], trace[int(0.3/dt):], dt, 0.05
            )
            cells = int(R_max / dr) + 1
            rows.append({
                "Method": "PDE", "Closure": closure, "resolution": dr,
                "replicate": 0, "runtime_s": diag["runtime_s"],
                "state_memory_mb": 8.0 * arrays_per_cell * cells / 1e6,
                **metrics, **diag,
            })
    pd.DataFrame(rows).to_csv(outdir / "accuracy_cost_frontier.csv", index=False)
    pd.DataFrame({
        "reference_N": [100000], "reference_replicates": [2],
        "mean_reference_runtime_s": [np.mean(reference_runtime)],
    }).to_csv(outdir / "accuracy_cost_reference.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--outdir", default="results/extended_revision")
    parser.add_argument("--section", choices=["all", "cumulants", "baseline",
                                               "cross_chi", "adaptation",
                                               "joint_grid", "joint_timestep",
                                               "transient", "accuracy_cost"],
                        default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    with open(outdir / "study_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    jobs = {
        "cumulants": run_joint_cumulants,
        "baseline": run_joint_baseline,
        "cross_chi": run_cross_chi,
        "adaptation": run_adaptation_sweep,
        "joint_grid": run_joint_grid,
        "joint_timestep": run_joint_timestep,
        "transient": run_transient,
        "accuracy_cost": run_accuracy_cost,
    }
    selected = jobs if args.section == "all" else {args.section: jobs[args.section]}
    for name, job in selected.items():
        print(f"[extended study] starting {name}", flush=True)
        job(cfg, outdir)
        print(f"[extended study] completed {name}", flush=True)


if __name__ == "__main__":
    main()
