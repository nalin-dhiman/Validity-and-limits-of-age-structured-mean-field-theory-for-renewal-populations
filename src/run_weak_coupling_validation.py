import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from metrics import ActivityMetrics
from simulate_population import run_population_sim


def parse_j_values(raw: str | None) -> list[float]:
    if raw is None or raw.strip() == "":
        return [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    return [float(tok) for tok in raw.split(",")]


def format_j_token(value: float) -> str:
    """Return a stable filename token without rounding distinct J values together."""
    token = np.format_float_positional(float(value), trim="-", precision=12)
    return token.replace("-", "m").replace(".", "p")


def run_validation(config_path, out_dir, duration=3.0, N=2000, c=0.0, j_vals=None):
    with open(config_path, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    os.makedirs(out_dir, exist_ok=True)

    # Extend the validated window beyond the original weak-coupling set so the
    # revised manuscript can document nonzero-J stability more convincingly.
    if j_vals is None:
        j_vals = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    seeds = [7, 11, 13]
    rows = []

    for J in j_vals:
        for seed in seeds:
            cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
            cfg["duration"] = float(duration)
            cfg["population"]["N"] = int(N)
            cfg["population"]["shared_noise_fraction"] = float(c)
            cfg["coupling"]["J"] = float(J)
            cfg["defaults"]["seed"] = int(seed)
            cfg["pde"]["R_max"] = "auto"

            j_token = format_j_token(J)
            outfile = os.path.join(out_dir, f"weak_J{j_token}_s{seed}.npz")
            run_population_sim(cfg, outfile, online_pde=True)

            data = np.load(outfile, allow_pickle=True)
            A_mc = data["A"]
            A_pde = data["A_pde"]
            pde_metrics = data["pde_metrics"].item()

            dt = cfg["defaults"]["dt"]
            burn_steps = int(0.5 / dt)
            nrmse, rmse = ActivityMetrics.compute_nrmse(
                A_mc[burn_steps:], A_pde[burn_steps:], dt, smooth_window=0.05
            )

            rows.append(
                {
                    "J": float(J),
                    "Seed": int(seed),
                    "Duration": float(duration),
                    "N": int(N),
                    "SharedNoise": float(c),
                    "RMSE": float(rmse),
                    "NRMSE": float(nrmse),
                    "MassMin": float(pde_metrics["mass_min"]),
                    "MassFinal": float(pde_metrics["mass_final"]),
                    "TailMassMax": float(pde_metrics["tail_mass_max"]),
                    "CumLeakFinal": float(pde_metrics["cum_leak_final"]),
                    "R_max": float(pde_metrics["R_max"]),
                }
            )

    df = pd.DataFrame(rows).sort_values(["J", "Seed"]).reset_index(drop=True)
    out_csv = os.path.join(out_dir, "weak_coupling_validation.csv")
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("J")
        .agg(
            NRMSE_mean=("NRMSE", "mean"),
            NRMSE_std=("NRMSE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MassMin_mean=("MassMin", "mean"),
            MassMin_min=("MassMin", "min"),
            TailMassMax_max=("TailMassMax", "max"),
        )
        .reset_index()
    )
    out_summary = os.path.join(out_dir, "weak_coupling_summary.csv")
    summary.to_csv(out_summary, index=False)

    print(f"Saved {out_csv}")
    print(f"Saved {out_summary}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out_dir", default="results/weak_coupling")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--shared-noise", type=float, default=0.0)
    parser.add_argument(
        "--j_values",
        default=None,
        help="Comma-separated list of J values. Defaults to 0,1e-4,2e-4,5e-4,1e-3,2e-3.",
    )
    args = parser.parse_args()

    run_validation(
        config_path=args.config,
        out_dir=args.out_dir,
        duration=args.duration,
        N=args.N,
        c=args.shared_noise,
        j_vals=parse_j_values(args.j_values),
    )
