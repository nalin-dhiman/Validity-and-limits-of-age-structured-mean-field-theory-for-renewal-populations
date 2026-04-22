#!/usr/bin/env python3
"""
Generate the main-text and supplementary figure PDFs from the CSV summaries in ../data.

Usage
-----
python3 make_paper_figures.py

This script is intentionally lightweight: it regenerates the figure PDFs
from the included summary CSV files, without rerunning the full simulation pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def fit_alpha_pure_power(df: pd.DataFrame, closure: str, metric_col: str) -> float:
    """Fit alpha from y ~ N^{-alpha} using a pure power law (log-log linear fit)."""
    sub = df[df["Closure"] == closure].groupby("N")[metric_col].mean()
    Ns = sub.index.values.astype(float)
    ys = sub.values.astype(float)
    mask = np.isfinite(ys) & (ys > 0)
    Ns, ys = Ns[mask], ys[mask]
    if len(Ns) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(Ns), np.log(ys), 1)
    return -float(slope)


def add_figure_legend(fig, handles, labels, *, ncol: int, y: float = 0.98, fontsize: int = 8) -> None:
    """Place a shared legend above the axes so the plotting area stays clean."""
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        columnspacing=1.2,
        handletextpad=0.6,
    )


def add_axis_legend_outside(
    ax,
    *,
    anchor=(1.02, 1.0),
    loc: str = "upper left",
    fontsize: int = 7,
    ncol: int = 1,
) -> None:
    """Place a single-axis legend outside the plotting area."""
    ax.legend(
        loc=loc,
        bbox_to_anchor=anchor,
        borderaxespad=0.0,
        frameon=False,
        fontsize=fontsize,
        ncol=ncol,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    paper_fig_dir = root / "figures" / "main"
    supp_fig_dir = root / "figures" / "supplementary"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    supp_fig_dir.mkdir(parents=True, exist_ok=True)

    bias_df = pd.read_csv(data_dir / "bias_fix_summary_seeds.csv")
    scaling_df = pd.read_csv(data_dir / "scaling_compare.csv")
    variance_df = pd.read_csv(data_dir / "variance_compare.csv")
    weak_df = pd.read_csv(data_dir / "weak_coupling_summary.csv")
    softplus_path = data_dir / "softplus_robustness_summary.csv"
    softplus_df = pd.read_csv(softplus_path) if softplus_path.exists() else None

    closure_order = ["age_only", "age_adapt_no_jensen", "age_adapt_jensen"]
    scaling_order = ["age_adapt_jensen", "age_adapt_no_jensen", "age_only"]
    scaling_colors = {
        "age_adapt_jensen": "C0",
        "age_adapt_no_jensen": "C1",
        "age_only": "C2",
    }
    scaling_markers = {
        "age_adapt_jensen": "o",
        "age_adapt_no_jensen": "s",
        "age_only": "D",
    }
    scaling_linestyles = {
        "age_adapt_jensen": "-",
        "age_adapt_no_jensen": "--",
        "age_only": "-.",
    }
    scaling_fillstyles = {
        "age_adapt_jensen": "none",
        "age_adapt_no_jensen": "full",
        "age_only": "none",
    }

    # -----------------------------
    # Fig 1 (main): Bias + NRMSE
    # -----------------------------
    def save_bias_and_nrmse(outpath: Path) -> None:
        x = np.arange(len(closure_order))
        width = 0.35

        nrmse20 = bias_df.groupby("Closure")["NRMSE_20ms"].mean().reindex(closure_order)
        nrmse20_std = bias_df.groupby("Closure")["NRMSE_20ms"].std().reindex(closure_order)
        nrmse50 = bias_df.groupby("Closure")["NRMSE_50ms"].mean().reindex(closure_order)
        nrmse50_std = bias_df.groupby("Closure")["NRMSE_50ms"].std().reindex(closure_order)

        bias20 = bias_df.groupby("Closure")["Bias_20ms"].mean().reindex(closure_order)
        bias20_std = bias_df.groupby("Closure")["Bias_20ms"].std().reindex(closure_order)
        bias50 = bias_df.groupby("Closure")["Bias_50ms"].mean().reindex(closure_order)
        bias50_std = bias_df.groupby("Closure")["Bias_50ms"].std().reindex(closure_order)

        fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.3))

        axs[0].bar(x - width / 2, nrmse20.values, width, yerr=nrmse20_std.values, capsize=3, label="20 ms")
        axs[0].bar(x + width / 2, nrmse50.values, width, yerr=nrmse50_std.values, capsize=3, label="50 ms")
        axs[0].set_yscale("log")
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(closure_order, rotation=20, ha="right")
        axs[0].set_ylabel("NRMSE")
        axs[0].set_title("(a) Normalized error")

        axs[1].bar(x - width / 2, bias20.values, width, yerr=bias20_std.values, capsize=3, label="20 ms")
        axs[1].bar(x + width / 2, bias50.values, width, yerr=bias50_std.values, capsize=3, label="50 ms")
        axs[1].axhline(0, lw=0.8, color="k", alpha=0.5)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(closure_order, rotation=20, ha="right")
        axs[1].set_ylabel("Bias (Hz)")
        axs[1].set_title("(b) Bias")

        handles, labels = axs[0].get_legend_handles_labels()
        add_figure_legend(fig, handles, labels, ncol=2, y=0.995, fontsize=8)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    save_bias_and_nrmse(paper_fig_dir / "Fig1_BiasAndNRMSE.pdf")

    # -----------------------------
    # Fig 2 (main): variance vs age with sample counts
    # -----------------------------
    def save_variance_check(outpath: Path) -> None:
        df = variance_df.copy()
        df["count"] = df["count"].fillna(0).astype(int)
        df["var_se"] = df["mc_variance"] * np.sqrt(2.0 / np.maximum(df["count"] - 1, 1))

        # Plot only bins with enough samples in the main comparison panel.
        valid = df["count"] >= 200
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(4.8, 5.2),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.2]},
        )

        ax = axs[0]
        ax.plot(df["age_center"], df["theory_variance"], color="k", lw=1.8, label="OU prediction")
        ax.fill_between(
            df.loc[valid, "age_center"].values,
            np.maximum(
                df.loc[valid, "mc_variance"].values - 1.96 * df.loc[valid, "var_se"].values,
                0.0,
            ),
            df.loc[valid, "mc_variance"].values + 1.96 * df.loc[valid, "var_se"].values,
            color="C0",
            alpha=0.18,
            linewidth=0,
            label="MC 95% CI",
        )
        ax.plot(
            df.loc[valid, "age_center"],
            df.loc[valid, "mc_variance"],
            "o",
            ms=3.0,
            color="C0",
            label="MC variance",
        )
        ax.axvspan(0.0, 0.02, color="0.85", alpha=0.8)
        ax.set_ylabel(r"$\mathrm{Var}(V \mid r)$")
        ax.set_title("Variance profile used in the Jensen correction")

        ax = axs[1]
        ax.plot(df["age_center"], df["count"], color="C1", lw=1.5)
        ax.axhline(200, color="0.4", ls="--", lw=0.9)
        ax.set_ylabel("count")
        ax.set_xlabel("Age $r$ (s)")
        ax.set_ylim(bottom=0)

        handles, labels = axs[0].get_legend_handles_labels()
        handles.append(Patch(facecolor="0.85", edgecolor="none", alpha=0.8))
        labels.append("boundary layer")
        add_figure_legend(fig, handles, labels, ncol=4, y=0.995, fontsize=7)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    save_variance_check(paper_fig_dir / "Fig2_VarianceCheck.pdf")

    # -----------------------------
    # Fig 3 (main): scaling of time-series error
    # -----------------------------
    def save_scaling_time_series_error(outpath: Path) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))

        ref_x = np.array(sorted(scaling_df["N"].unique()), dtype=float)
        ref_scale = np.median(
            scaling_df.groupby("N")["NRMSE_50ms"].mean().values * np.sqrt(ref_x)
        )
        ref_y = ref_scale * ref_x ** (-0.5)
        ax.loglog(
            ref_x,
            ref_y,
            linestyle=":",
            color="red",
            linewidth=1.0,
            label=r"$N^{-1/2}$ reference",
        )

        for closure in scaling_order:
            sub = (
                scaling_df[scaling_df["Closure"] == closure]
                .groupby("N")["NRMSE_50ms"]
                .agg(["mean", "std"])
                .sort_index()
            )
            alpha = fit_alpha_pure_power(scaling_df, closure, "NRMSE_50ms")
            label = f"{closure} (\u03b1={alpha:.2f})"
            ax.errorbar(
                sub.index.values,
                sub["mean"].values,
                yerr=sub["std"].fillna(0.0).values,
                marker=scaling_markers[closure],
                color=scaling_colors[closure],
                linestyle=scaling_linestyles[closure],
                linewidth=1.4,
                markersize=6.5,
                markerfacecolor="white" if scaling_fillstyles[closure] == "none" else scaling_colors[closure],
                markeredgecolor=scaling_colors[closure],
                markeredgewidth=1.5,
                capsize=3,
                elinewidth=1.0,
                alpha=0.95,
                zorder=3,
                label=label,
            )

        ax.set_xlabel("Population size $N$")
        ax.set_ylabel(r"Centered NRMSE ($\tau = 50$ ms)")
        ax.set_title("Finite-size scaling of time-series error")
        ax.set_xlim(ref_x.min() * 0.8, ref_x.max() * 1.2)
        ax.grid(True, which="both", alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        add_figure_legend(fig, handles, labels, ncol=2, y=0.995, fontsize=7)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    save_scaling_time_series_error(paper_fig_dir / "Fig3_ScalingTimeSeriesError.pdf")

    # -----------------------------
    # Fig 4 (main): weak-coupling validation
    # -----------------------------
    def save_weak_coupling_validation(outpath: Path) -> None:
        x = weak_df["J"].values * 1e4
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.3))

        axs[0].errorbar(
            x,
            weak_df["NRMSE_mean"],
            yerr=weak_df["NRMSE_std"].fillna(0.0),
            marker="o",
            color="C0",
            capsize=3,
        )
        axs[0].set_xlabel(r"$J \times 10^{4}$")
        axs[0].set_ylabel("NRMSE")
        axs[0].set_title("(a) MC vs PDE error")

        axs[1].plot(x, weak_df["MassMin_mean"], marker="o", color="C1", label="Mean minimum mass")
        axs[1].plot(x, weak_df["MassMin_min"], marker="s", color="C3", label="Worst seed")
        axs[1].axhline(0.95, color="0.4", ls="--", lw=0.9, label="hard guard")
        axs[1].set_xlabel(r"$J \times 10^{4}$")
        axs[1].set_ylabel(r"$\min_t M(t)$")
        axs[1].set_ylim(0.94, 1.005)
        axs[1].set_title("(b) Probability mass")

        handles, labels = axs[1].get_legend_handles_labels()
        add_figure_legend(fig, handles, labels, ncol=3, y=0.995, fontsize=7)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    save_weak_coupling_validation(paper_fig_dir / "Fig4_WeakCouplingValidation.pdf")

    # -----------------------------
    # Supplementary: scaling of mean error
    # -----------------------------
    def save_scaling_mean_error(outpath: Path) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        for cl in closure_order:
            sub = (
                scaling_df[scaling_df["Closure"] == cl]
                .groupby("N")["NRMSE_mean_20ms"]
                .mean()
                .sort_index()
            )
            ax.plot(sub.index.values, sub.values, marker="o", label=cl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Population size $N$")
        ax.set_ylabel("Mean-rate NRMSE (20 ms)")
        ax.set_title("Scaling of mean-rate error")
        add_axis_legend_outside(ax)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    save_scaling_mean_error(supp_fig_dir / "FigS1_ScalingMeanError.pdf")

    # -----------------------------
    # Supplementary: alpha sensitivity (pure power law)
    # -----------------------------
    def save_alpha_sensitivity(outpath: Path) -> None:
        windows = [20, 50, 100]
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        for cl in closure_order:
            alphas = [fit_alpha_pure_power(scaling_df, cl, f"NRMSE_{w}ms") for w in windows]
            ax.plot(windows, alphas, marker="o", linestyle="-", label=cl)
        ax.set_xlabel("Smoothing window (ms)")
        ax.set_ylabel(r"Fitted exponent $\alpha$ (NRMSE $\sim N^{-\alpha}$)")
        ax.set_title("Scaling exponent vs smoothing")
        ax.set_ylim(0, 0.6)
        add_axis_legend_outside(ax)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    save_alpha_sensitivity(supp_fig_dir / "FigS2_AlphaSensitivity.pdf")

    # -----------------------------
    # Supplementary: seed variation plots
    # -----------------------------
    def save_boxplot(outpath: Path, metric: str, ylabel: str, logy: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        data = [bias_df[bias_df["Closure"] == cl][metric].values for cl in closure_order]
        ax.boxplot(data, tick_labels=closure_order, showmeans=True)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Seed-to-seed variation: {metric}")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

    save_boxplot(supp_fig_dir / "FigS3_BiasSeedVariation_NRMSE50.pdf", "NRMSE_50ms", "NRMSE (50 ms)", logy=True)
    save_boxplot(supp_fig_dir / "FigS4_BiasSeedVariation_Bias20.pdf", "Bias_20ms", "Bias (Hz, 20 ms)", logy=False)

    # -----------------------------
    # Supplementary: softplus robustness
    # -----------------------------
    if softplus_df is not None:
        def save_softplus_robustness(outpath: Path) -> None:
            x = softplus_df["J"].values * 1e4
            fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.3))

            axs[0].errorbar(
                x,
                softplus_df["NRMSE_mean"],
                yerr=softplus_df["NRMSE_std"].fillna(0.0),
                marker="o",
                color="C0",
                capsize=3,
            )
            axs[0].set_xlabel(r"$J \times 10^{4}$")
            axs[0].set_ylabel("NRMSE")
            axs[0].set_title("(a) MC vs PDE error")

            axs[1].plot(x, softplus_df["MassMin_mean"], marker="o", color="C1", label="Mean minimum mass")
            axs[1].plot(x, softplus_df["MassMin_min"], marker="s", color="C3", label="Worst seed")
            axs[1].axhline(0.95, color="0.4", ls="--", lw=0.9, label="hard guard")
            axs[1].set_xlabel(r"$J \times 10^{4}$")
            axs[1].set_ylabel(r"$\min_t M(t)$")
            axs[1].set_ylim(0.94, 1.005)
            axs[1].set_title("(b) Probability mass")

            handles, labels = axs[1].get_legend_handles_labels()
            add_figure_legend(fig, handles, labels, ncol=3, y=0.995, fontsize=7)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
            fig.savefig(outpath, bbox_inches="tight")
            plt.close(fig)

        save_softplus_robustness(supp_fig_dir / "FigS5_SoftplusRobustness.pdf")

    print("Wrote figures to:")
    print("  ", paper_fig_dir)
    print("  ", supp_fig_dir)


if __name__ == "__main__":
    main()
