import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from simulate_population import run_population_sim


def measure_activity_stats(config, outfile, burn_in=0.2, smooth_window=0.05):
    run_population_sim(config, outfile, online_pde=False)
    data = np.load(outfile, allow_pickle=True)
    activity = data["A"]
    dt = config["defaults"]["dt"]
    burn_steps = int(burn_in / dt)
    activity = activity[burn_steps:]
    activity_smooth = gaussian_filter1d(activity, sigma=smooth_window / dt)
    return float(activity.mean()), float(activity_smooth.std())


def build_softplus_config(base_cfg, theta0):
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["spike_gen"]["type"] = "softplus"
    cfg["spike_gen"]["theta_0"] = float(theta0)
    cfg["pde"]["use_jensen"] = False
    return cfg


def calibrate_softplus(
    base_config_path,
    out_dir,
    duration=1.0,
    N=1000,
    seeds=None,
    theta0_candidates=None,
):
    if seeds is None:
        seeds = [7, 11, 13]
    if theta0_candidates is None:
        theta0_candidates = [7.5, 8.0, 8.5, 9.0]

    with open(base_config_path, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for seed in seeds:
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
        cfg["duration"] = float(duration)
        cfg["population"]["N"] = int(N)
        cfg["defaults"]["seed"] = int(seed)
        mean_rate, smooth_std = measure_activity_stats(
            cfg, os.path.join(out_dir, f"exp_target_s{seed}.npz")
        )
        rows.append(
            {
                "kind": "exp_target",
                "theta0": float(cfg["spike_gen"]["theta_0"]),
                "seed": int(seed),
                "mean_rate": mean_rate,
                "smooth_std": smooth_std,
            }
        )

    for theta0 in theta0_candidates:
        for seed in seeds:
            cfg = build_softplus_config(base_cfg, theta0)
            cfg["duration"] = float(duration)
            cfg["population"]["N"] = int(N)
            cfg["defaults"]["seed"] = int(seed)
            mean_rate, smooth_std = measure_activity_stats(
                cfg, os.path.join(out_dir, f"softplus_t{theta0:.1f}_s{seed}.npz")
            )
            rows.append(
                {
                    "kind": "softplus",
                    "theta0": float(theta0),
                    "seed": int(seed),
                    "mean_rate": mean_rate,
                    "smooth_std": smooth_std,
                }
            )

    df = pd.DataFrame(rows)
    target = df[df["kind"] == "exp_target"][["mean_rate", "smooth_std"]].mean()

    summary = (
        df[df["kind"] == "softplus"]
        .groupby("theta0")
        .agg(
            mean_rate_mean=("mean_rate", "mean"),
            mean_rate_std=("mean_rate", "std"),
            smooth_std_mean=("smooth_std", "mean"),
            smooth_std_std=("smooth_std", "std"),
        )
        .reset_index()
    )
    summary["score"] = (
        ((summary["mean_rate_mean"] - target["mean_rate"]) / target["mean_rate"]) ** 2
        + ((summary["smooth_std_mean"] - target["smooth_std"]) / (target["smooth_std"] + 1e-12)) ** 2
    )
    summary["target_mean_rate"] = float(target["mean_rate"])
    summary["target_smooth_std"] = float(target["smooth_std"])

    best_row = summary.sort_values("score").iloc[0]
    best_theta0 = float(best_row["theta0"])
    best_cfg = build_softplus_config(base_cfg, best_theta0)

    raw_path = os.path.join(out_dir, "softplus_calibration_raw.csv")
    summary_path = os.path.join(out_dir, "softplus_calibration_summary.csv")
    config_path = os.path.join("configs", "softplus_calibrated.yaml")
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(best_cfg, fh, sort_keys=False)

    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Wrote calibrated config to {config_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--out_dir", default="results/softplus_calibration")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=1000)
    args = parser.parse_args()

    calibrate_softplus(
        base_config_path=args.base_config,
        out_dir=args.out_dir,
        duration=args.duration,
        N=args.N,
    )
