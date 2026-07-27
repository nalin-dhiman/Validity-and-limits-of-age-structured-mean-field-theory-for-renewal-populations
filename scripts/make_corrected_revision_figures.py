#!/usr/bin/env python3
"""Create the corrected main and supplementary figures from audited CSV data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]

# Prefer newly generated results, while allowing the tracked processed data to
# regenerate every figure immediately after cloning the repository.
DATA = ROOT / "results" / "corrected_revision"
if not DATA.exists():
    DATA = ROOT / "data" / "corrected"
EXT = ROOT / "results" / "extended_revision"
if not EXT.exists():
    EXT = ROOT / "data" / "extended"
JOINT = ROOT / "results" / "joint_revision"
if not JOINT.exists():
    JOINT = ROOT / "data" / "joint"

MAIN = ROOT / "figures" / "main"
SUPP = ROOT / "figures" / "supplementary"
MAIN.mkdir(parents=True, exist_ok=True)
SUPP.mkdir(parents=True, exist_ok=True)

LABELS = {
    "adaptation_neglecting": "adaptation-neglecting",
    "mean_only": "mean-only",
    "variance_corrected": "variance-corrected",
    "joint_gaussian": "dynamic joint-Gaussian",
}
COLORS = {
    "adaptation_neglecting": "#777777",
    "mean_only": "#d95f02",
    "variance_corrected": "#1b9e77",
    "joint_gaussian": "#7570b3",
}


def finish(fig, path, top=0.96):
    """Apply layout while reserving space above the axes for external legends."""
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def external_legend(fig, axes, ncol=3, fontsize=7, top=0.82):
    """Collect unique entries and place one legend outside all plotting axes."""
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=ncol, frameon=False, fontsize=fontsize,
               columnspacing=1.2, handlelength=2.2)
    return top


def bias_figure():
    joint_path = JOINT / "joint_closure_baseline.csv"
    if joint_path.exists():
        df = pd.read_csv(joint_path)
        order = ["mean_only", "variance_corrected", "joint_gaussian"]
    else:
        df = pd.read_csv(DATA / "bias_corrected.csv")
        order = ["adaptation_neglecting", "mean_only", "variance_corrected"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, metric, ylabel in [
        (axes[0], "bias_hz_50ms", "Bias (Hz)"),
        (axes[1], "rmse_hz_50ms", "RMSE (Hz)"),
    ]:
        means = df.groupby("Closure")[metric].mean().reindex(order)
        stds = df.groupby("Closure")[metric].std().reindex(order)
        ax.bar(x, means, yerr=stds, capsize=3,
               color=[COLORS[v] for v in order], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x, [LABELS[v] for v in order], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", lw=0.7)
    axes[0].set_title("(a) Signed mean-rate error")
    axes[1].set_title("(b) Closure hierarchy")
    finish(fig, MAIN / "Fig1_CorrectedBias.pdf")


def jensen_figure():
    df = pd.read_csv(DATA / "jensen_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for variant in ["mean_only", "variance_corrected"]:
        sub = df[df.Closure == variant].groupby("chi_inf").agg(
            bias=("bias_hz_50ms", "mean"), bias_sd=("bias_hz_50ms", "std"),
            rmse=("rmse_hz_50ms", "mean"), rmse_sd=("rmse_hz_50ms", "std"),
        )
        axes[0].errorbar(sub.index, sub.bias, yerr=sub.bias_sd, marker="o",
                         capsize=3, color=COLORS[variant], label=LABELS[variant])
        axes[1].errorbar(sub.index, sub.rmse, yerr=sub.rmse_sd, marker="o",
                         capsize=3, color=COLORS[variant], label=LABELS[variant])
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylabel("Bias (Hz)")
    axes[1].set_ylabel("RMSE (Hz)")
    for ax in axes:
        ax.set_xlabel(r"Convexity load $\chi_\infty$")
        ax.grid(alpha=0.25)
    axes[0].set_title("(a) Bias growth")
    axes[1].set_title("(b) Correction benefit and limit")
    top = external_legend(fig, axes, ncol=2, fontsize=8, top=0.84)
    finish(fig, MAIN / "Fig2_JensenValidity.pdf", top=top)


def scaling_figure():
    path = JOINT / "scaling_corrected.csv"
    df = pd.read_csv(path if path.exists() else DATA / "scaling_corrected.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    order = [v for v in ["adaptation_neglecting", "mean_only", "variance_corrected",
                          "joint_gaussian"] if v in set(df.Closure)]
    for variant in order:
        sub = df[df.Closure == variant].groupby("N").agg(
            total=("rmse_hz_50ms", "mean"), total_sd=("rmse_hz_50ms", "std"),
            centered=("crmse_hz_50ms", "mean"), centered_sd=("crmse_hz_50ms", "std"),
        )
        axes[0].errorbar(sub.index, sub.total, yerr=sub.total_sd, marker="o",
                         capsize=2, color=COLORS[variant], label=LABELS[variant])
        axes[1].errorbar(sub.index, sub.centered, yerr=sub.centered_sd, marker="o",
                         capsize=2, color=COLORS[variant], label=LABELS[variant])
    refN = np.array(sorted(df.N.unique()), dtype=float)
    reference_variant = "joint_gaussian" if "joint_gaussian" in order else "variance_corrected"
    centered = (df[df.Closure == reference_variant].groupby("N")
                ["crmse_hz_50ms"].mean())
    ref = centered.iloc[0] * np.sqrt(refN[0] / refN)
    axes[1].plot(refN, ref, "k--", lw=1, label=r"$N^{-1/2}$ reference")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Population size $N$")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Total RMSE (Hz)")
    axes[1].set_ylabel("Centered RMSE (Hz)")
    axes[0].set_title("(a) Deterministic error floor")
    axes[1].set_title("(b) Finite-size fluctuations")
    top = external_legend(fig, axes, ncol=4, fontsize=7, top=0.84)
    finish(fig, MAIN / "Fig3_ErrorDecomposition.pdf", top=top)


def coupling_figure():
    path = JOINT / "coupling_closed_loop.csv"
    df = pd.read_csv(path if path.exists() else DATA / "coupling_closed_loop.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    if "Closure" not in df:
        df["Closure"] = "variance_corrected"
    for variant in ["variance_corrected", "joint_gaussian"]:
        part = df[df.Closure == variant]
        if part.empty:
            continue
        sub = part.groupby("g").agg(
            rmse=("rmse_hz_50ms", "mean"), rmse_sd=("rmse_hz_50ms", "std"),
            nrmse=("nrmse_50ms", "mean"), nrmse_sd=("nrmse_50ms", "std"),
        )
        axes[0].errorbar(sub.index, sub.rmse, yerr=sub.rmse_sd, marker="o",
                         capsize=3, color=COLORS[variant], label=LABELS[variant])
        axes[1].errorbar(sub.index, sub.nrmse, yerr=sub.nrmse_sd, marker="o",
                         capsize=3, color=COLORS[variant], label=LABELS[variant])
    axes[0].set_ylabel("Closed-loop RMSE (Hz)")
    axes[0].set_title("(a) Predictive error")
    axes[1].axhline(1.0, color="0.35", lw=0.8, ls="--")
    axes[1].set_ylabel("NRMSE")
    axes[1].set_title("(b) Error relative to MC variability")
    for ax in axes:
        ax.set_xlabel(r"Normalized feedback $g=\tau_m J A_0/V_{\rm ref}$")
        ax.grid(alpha=0.25)
    top = external_legend(fig, axes, ncol=2, fontsize=8, top=0.84)
    finish(fig, MAIN / "Fig4_ClosedLoopCoupling.pdf", top=top)


def performance_figure():
    df = pd.read_csv(DATA / "performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    axes[0].loglog(df.N, df.MC_runtime_s, "o-", label="microscopic MC")
    axes[0].loglog(df.N, df.PDE_runtime_s, "s--", label="PDE closure")
    axes[0].set_ylabel("Wall time for 1 s model time (s)")
    axes[0].set_title("(a) Computational cost")
    axes[1].semilogx(df.N, df.speedup_MC_over_PDE, "o-", color="#7570b3")
    axes[1].axhline(1, color="black", lw=0.8, ls="--")
    axes[1].set_ylabel("Speedup (MC time / PDE time)")
    axes[1].set_title("(b) Measured crossover")
    for ax in axes:
        ax.set_xlabel("Population size $N$")
        ax.grid(alpha=0.25, which="both")
    top = external_legend(fig, axes, ncol=2, fontsize=8, top=0.84)
    finish(fig, MAIN / "Fig5_Performance.pdf", top=top)


def variance_figure():
    path = DATA / "variance_compare.csv"
    if not path.exists():
        path = ROOT / "data" / "variance_compare.csv"
    if not path.exists():
        path = ROOT / "revision_package" / "data" / "variance_compare.csv"
    df = pd.read_csv(path)
    df["count"] = df["count"].fillna(0)
    valid = df["count"] >= 200
    V_ref_sq = 1e-6
    se = df["mc_variance"] * np.sqrt(2.0 / np.maximum(df["count"] - 1, 1))
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2]})
    axes[0].plot(df.age_center, df.theory_variance / V_ref_sq, "k-", label="OU prediction")
    axes[0].fill_between(df.loc[valid, "age_center"],
                         np.maximum((df.loc[valid, "mc_variance"] - 1.96 * se[valid]) / V_ref_sq, 0),
                         (df.loc[valid, "mc_variance"] + 1.96 * se[valid]) / V_ref_sq,
                         alpha=0.2, label="MC 95% CI")
    axes[0].plot(df.loc[valid, "age_center"],
                 df.loc[valid, "mc_variance"] / V_ref_sq, "o", ms=3, label="MC")
    axes[0].set_ylabel(r"$\mathrm{Var}(V\mid r)/V_{\rm ref}^2$")
    axes[1].plot(df.age_center, df["count"], color="#d95f02")
    axes[1].axhline(200, color="black", ls="--", lw=0.8, label="reporting threshold")
    axes[1].set_ylabel("count")
    axes[1].set_xlabel("Age $r$ (s)")
    top = external_legend(fig, axes, ncol=4, fontsize=7, top=0.86)
    finish(fig, SUPP / "FigS1_NormalizedVariance.pdf", top=top)


def convergence_figure():
    df = pd.read_csv(DATA / "grid_convergence.csv")
    fig, ax = plt.subplots(figsize=(4.5, 3.1))
    ax.loglog(df.dr.iloc[:-1], df.rmse_vs_finest_hz.iloc[:-1], "o-")
    ax.set_xlabel(r"Age-grid spacing $\Delta r$ (s)")
    ax.set_ylabel("RMSE relative to finest grid (Hz)")
    ax.grid(alpha=0.25, which="both")
    finish(fig, SUPP / "FigS2_GridConvergence.pdf")


def cross_parameter_validity_figure():
    source = JOINT if (JOINT / "cross_parameter_chi.csv").exists() else EXT
    df = pd.read_csv(source / "cross_parameter_chi.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for variant in ["mean_only", "variance_corrected", "joint_gaussian"]:
        if variant not in set(df.Closure):
            continue
        sub = df[df.Closure == variant].groupby("chi_target").rmse_hz_50ms.agg(["mean", "std"])
        axes[0].errorbar(sub.index, sub["mean"], yerr=sub["std"], marker="o",
                         capsize=2, color=COLORS[variant], label=LABELS[variant])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE (Hz)")
    axes[0].set_xlabel(r"Convexity load $\chi_\infty$")
    axes[0].set_title("(a) Gain--noise families")

    adaptation_path = JOINT / "adaptation_strength_sweep.csv"
    if adaptation_path.exists():
        adaptation = pd.read_csv(adaptation_path)
        for variant in ["mean_only", "variance_corrected", "joint_gaussian"]:
            sub = adaptation[adaptation.Closure == variant].groupby("kappa").rmse_hz_50ms.agg(["mean", "std"])
            axes[1].errorbar(sub.index, sub["mean"], yerr=sub["std"], marker="o",
                             capsize=2, color=COLORS[variant], label=LABELS[variant])
        axes[1].set_yscale("log")
        axes[1].set_ylabel("RMSE (Hz)")
        axes[1].set_xlabel(r"Adaptation strength $\kappa$ (s$^{-1}$)")
        axes[1].set_title("(b) Adaptation generalization")
    else:
        axes[1].axis("off")
    for ax in axes:
        ax.grid(alpha=0.25)
    top = external_legend(fig, axes, ncol=3, fontsize=7, top=0.82)
    finish(fig, MAIN / "Fig2_CrossParameterValidity.pdf", top=top)


def joint_cumulant_figure():
    source = JOINT if (JOINT / "joint_cumulant_diagnostic.csv").exists() else EXT
    df = pd.read_csv(source / "joint_cumulant_diagnostic.csv")
    df = df[(df.age_s >= 0.0075) & (df.age_s <= 0.30) & (df["count"] > 10000)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    theta1, theta2 = 2000.0, -9800.0
    pde_terms = {
        "voltage": 0.5 * theta1 ** 2 * df.pde_var_v,
        "adaptation": 0.5 * theta2 ** 2 * df.pde_var_a,
        "covariance": theta1 * theta2 * df.pde_cov_va,
    }
    mc_terms = {"voltage": df.term_v, "adaptation": df.term_a,
                "covariance": df.term_cov}
    term_colors = {"voltage": "#1f77b4", "adaptation": "#ff7f0e",
                   "covariance": "#2ca02c"}
    for name in ["voltage", "adaptation", "covariance"]:
        axes[0].plot(df.age_s, mc_terms[name], color=term_colors[name],
                     label=f"MC {name}")
        axes[0].plot(df.age_s, pde_terms[name], color=term_colors[name], ls="--",
                     label=f"PDE {name}")
    axes[0].axhline(0, color="0.4", lw=0.7)
    axes[0].set_xlabel("Age $r$ (s)")
    axes[0].set_ylabel("Contribution to log hazard")
    axes[0].set_title("(a) Joint cumulant terms")
    axes[1].plot(df.age_s, df.hazard_empirical_hz, color="black", lw=2, label="empirical MC")
    axes[1].plot(df.age_s, df.hazard_mean_hz, "--", label="mean-only")
    axes[1].plot(df.age_s, df.hazard_voltage_only_hz, "-.", label="voltage variance")
    if "pde_hazard_hz" in df:
        axes[1].plot(df.age_s, df.pde_hazard_hz, color=COLORS["joint_gaussian"],
                     lw=1.8, label="predictive joint PDE")
    else:
        axes[1].plot(df.age_s, df.hazard_full_gaussian_hz, ":", lw=2,
                     label="joint Gaussian diagnostic")
    axes[1].set_xlabel("Age $r$ (s)")
    axes[1].set_ylabel("Conditional hazard (Hz)")
    axes[1].set_title("(b) Hazard reconstruction")
    for ax in axes:
        ax.grid(alpha=0.2)
    top = external_legend(fig, axes, ncol=5, fontsize=6.2, top=0.72)
    finish(fig, MAIN / "Fig3_JointCumulants.pdf", top=top)


def transient_figure():
    source = JOINT if (JOINT / "transient_step_traces.csv").exists() else EXT
    df = pd.read_csv(source / "transient_step_traces.csv")
    feat = pd.read_csv(source / "transient_step_features.csv")
    if "amplitude" in df:
        view = df[(df.amplitude == 0.003) & (df.time_s >= 1.5) & (df.time_s <= 3.5)]
    else:
        view = df[(df.time_s >= 1.5) & (df.time_s <= 3.5)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    axes[0].plot(view.time_s, view.mc_hz, color="black", label="MC")
    axes[0].plot(view.time_s, view.mean_only_hz, "--", color=COLORS["mean_only"], label="mean-only")
    axes[0].plot(view.time_s, view.variance_corrected_hz, color=COLORS["variance_corrected"], label="variance-corrected")
    if "joint_gaussian_hz" in view:
        axes[0].plot(view.time_s, view.joint_gaussian_hz,
                     color=COLORS["joint_gaussian"], label=LABELS["joint_gaussian"])
    axes[0].axvline(2.0, color="0.4", ls=":", label="step onset")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Population activity (Hz)")
    axes[0].set_title("(a) Step response")
    models = [m for m in ["mean_only", "variance_corrected", "joint_gaussian"]
              if m in set(feat.Model)]
    if "amplitude" in feat:
        for model in models:
            sub = feat[feat.Model == model].sort_values("amplitude")
            axes[1].plot(sub.amplitude, sub.rmse_hz, marker="o",
                         color=COLORS[model], label=LABELS[model])
        axes[1].set_xlabel(r"Step amplitude $\Delta u$ ($U\,\mathrm{s}^{-1}$)")
    else:
        x = np.arange(len(models))
        axes[1].bar(x, [feat.loc[feat.Model == m, "rmse_hz"].iloc[0] for m in models],
                    color=[COLORS[m] for m in models])
        axes[1].set_xticks(x, [LABELS[m] for m in models], rotation=15, ha="right")
    axes[1].set_ylabel("Total RMSE (Hz)")
    axes[1].set_title("(b) Generalization across step size")
    axes[1].grid(alpha=0.25)
    top = external_legend(fig, axes, ncol=4, fontsize=6.5, top=0.78)
    finish(fig, MAIN / "Fig4_TransientResponse.pdf", top=top)


def accuracy_cost_figure():
    source = JOINT if (JOINT / "accuracy_cost_frontier.csv").exists() else EXT
    df = pd.read_csv(source / "accuracy_cost_frontier.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    mc_markers = {1000.0: "o", 5000.0: "s", 20000.0: "^", 50000.0: "D"}
    mc_color = "#1f77b4"
    for ax, metric, title in [
        (axes[0], "rmse_hz", "(a) Total accuracy--cost"),
        (axes[1], "crmse_hz", "(b) Fluctuation accuracy--cost"),
    ]:
        mc = df[df.Method == "MC"].groupby("resolution").agg(
            runtime=("runtime_s", "mean"), runtime_sd=("runtime_s", "std"),
            error=(metric, "mean"), error_sd=(metric, "std"),
        )
        ax.plot(mc.runtime, mc.error, color=mc_color, lw=1.2)
        for resolution, row in mc.iterrows():
            ax.errorbar(row.runtime, row.error, xerr=row.runtime_sd, yerr=row.error_sd,
                        marker=mc_markers[float(resolution)], color=mc_color,
                        capsize=2, ls="none", label="_nolegend_")
        pde_all = df[df.Method == "PDE"].copy()
        if "Closure" not in pde_all:
            pde_all["Closure"] = "variance_corrected"
        for closure, marker, linestyle in [
            ("variance_corrected", "s", "--"), ("joint_gaussian", "P", "-.")
        ]:
            pde = pde_all[pde_all.Closure == closure].sort_values("resolution")
            if pde.empty:
                continue
            ax.plot(pde.runtime_s, pde[metric], marker=marker, ls=linestyle,
                    color=COLORS[closure], label=LABELS[closure])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall time for 2 s model time (s)")
        ax.set_ylabel("Reference error (Hz)")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
    handles = [Line2D([0], [0], color=mc_color, marker="o", label="microscopic MC")]
    if "Closure" in df:
        handles += [
            Line2D([0], [0], color=COLORS["variance_corrected"], marker="s", ls="--",
                   label="prescribed variance PDE"),
            Line2D([0], [0], color=COLORS["joint_gaussian"], marker="P", ls="-.",
                   label="dynamic joint PDE"),
        ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=7, columnspacing=1.2)
    finish(fig, MAIN / "Fig7_AccuracyCost.pdf", top=0.78)


if __name__ == "__main__":
    bias_figure()
    jensen_figure()
    scaling_figure()
    coupling_figure()
    performance_figure()
    variance_figure()
    convergence_figure()
    if EXT.exists():
        cross_parameter_validity_figure()
        joint_cumulant_figure()
        transient_figure()
        accuracy_cost_figure()
    print("Corrected revision figures written.")
