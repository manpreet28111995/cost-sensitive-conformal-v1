from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_coverage_bars(
    method_table: pd.DataFrame,
    output_path: str | Path,
    target_coverage: float = 0.90,
) -> None:
    """Grouped bars of coverage across all 4 conformal methods: Marginal, Mondrian, APS, RAPS."""
    fig, ax = plt.subplots(figsize=(10, 5))
    conformal_methods = ["marginal_cp", "mondrian_cp", "aps_cp", "raps_cp"]
    plot_frame = method_table[method_table["method"].isin(conformal_methods)].copy()
    
    # Calculate Overall Coverage, Class 0 Coverage, Class 1 Coverage
    grouped = (
        plot_frame.groupby("method")[["marginal_coverage", "coverage_class_0", "coverage_class_1"]]
        .mean()
        .rename(columns={
            "marginal_coverage": "Overall Coverage",
            "coverage_class_0": "Class 0 (Majority)",
            "coverage_class_1": "Class 1 (Minority)",
        })
    )
    # Reindex to enforce canonical method order
    grouped = grouped.reindex([m for m in conformal_methods if m in grouped.index])
    
    method_labels = {
        "marginal_cp": "Marginal CP",
        "mondrian_cp": "Mondrian CP",
        "aps_cp": "APS CP",
        "raps_cp": "RAPS CP",
    }
    grouped.index = [method_labels.get(m, m) for m in grouped.index]

    grouped.plot(kind="bar", ax=ax, color=["#2B5C8F", "#4C78A8", "#E45756"], rot=0, width=0.75)
    ax.axhline(
        target_coverage,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Target Coverage ({target_coverage:.0%})",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.set_xlabel("Conformal Prediction Method", fontsize=11)
    ax.set_title("Coverage Comparison: Overall vs. Class-Specific Across Conformal Methods", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cost_abstention_frontier(
    frontier: pd.DataFrame,
    output_path: str | Path,
    method_summary: pd.DataFrame | None = None,
) -> None:
    """Plot clear Pareto frontiers of decision cost vs abstention rate per method without raw overplotting."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 1. Aggregate frontier data by method and alpha step across datasets/models/seeds
    if not frontier.empty and "alpha" in frontier.columns:
        frontier_agg = (
            frontier.groupby(["method", "alpha"])[["abstention_rate", "mean_cost"]]
            .mean()
            .reset_index()
        )
        
        for method_key, label, color, marker, linestyle in [
            ("mondrian_cp", "Mondrian CP (Sweep)", "#C0392B", "o", "-"),
            ("marginal_cp", "Marginal CP (Sweep)", "#2B5C8F", "s", "--"),
        ]:
            sub = frontier_agg[frontier_agg["method"] == method_key].sort_values("abstention_rate")
            if not sub.empty:
                ax.plot(
                    sub["abstention_rate"],
                    sub["mean_cost"],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    linewidth=2.0,
                    markersize=5,
                    alpha=0.85,
                    label=label,
                )

    # 2. Overlay discrete operating points from method_summary if provided
    if method_summary is not None and not method_summary.empty:
        point_agg = (
            method_summary.groupby("method")[["abstention_rate_mean", "mean_cost_mean"]]
            .mean()
            .reset_index()
        )
        
        discrete_methods = {
            "cost_controlled_mondrian": ("Cost-Ctrl Mondrian (Optimal)", "#27AE60", "*", 14),
            "mondrian_cp": ("Mondrian CP (Target α=0.10)", "#E74C3C", "o", 9),
            "marginal_cp": ("Marginal CP (Target α=0.10)", "#3498DB", "s", 8),
            "bayes_cost_tuned": ("Bayes Cost Threshold", "#8E44AD", "D", 9),
            "cost_tuned_threshold": ("Bayes Cost Threshold", "#8E44AD", "D", 9),
            "confidence_rejector": ("Confidence Rejector", "#E67E22", "^", 9),
            "risk_controlled_rejector": ("Risk-Controlled Rejector", "#16A085", "v", 9),
            "aps_cp": ("APS CP", "#F39C12", "P", 8),
            "raps_cp": ("RAPS CP", "#D35400", "X", 8),
        }
        
        seen_labels = set()
        for _, row in point_agg.iterrows():
            m_name = row["method"]
            if m_name in discrete_methods:
                label, color, marker, msize = discrete_methods[m_name]
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                ax.plot(
                    row["abstention_rate_mean"],
                    row["mean_cost_mean"],
                    marker=marker,
                    color=color,
                    linestyle="None",
                    markersize=msize,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    label=label,
                    zorder=5,
                )

    ax.set_xlabel("Abstention Rate (Human Review Fraction)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expected Decision Cost", fontsize=11, fontweight="bold")
    ax.set_title("Cost-vs-Abstention Pareto Frontiers & Operating Points", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="gray", fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_deferral_sensitivity(sensitivity: pd.DataFrame, output_path: str | Path) -> None:
    """Deferral sensitivity plot aggregated per dataset to guarantee monotonic cost curves with break-even markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Aggregate per dataset and defer_cost to fix model-mixing non-monotonicity bug
    agg_df = (
        sensitivity.groupby(["dataset", "defer_cost"])[["mondrian_cost", "threshold_cost"]]
        .mean()
        .reset_index()
    )
    
    cmap = plt.get_cmap("tab20")
    datasets = agg_df["dataset"].unique()
    
    for idx, dataset in enumerate(datasets):
        group = agg_df[agg_df["dataset"] == dataset].sort_values("defer_cost")
        color = cmap(idx % 20)
        
        # Plot Mondrian Conformal Deferral (Solid line)
        ax.plot(
            group["defer_cost"],
            group["mondrian_cost"],
            "-o",
            color=color,
            linewidth=1.5,
            markersize=4,
            alpha=0.7,
        )
        # Plot Cost-Tuned Bayes Threshold (Dashed line)
        ax.plot(
            group["defer_cost"],
            group["threshold_cost"],
            "--",
            color=color,
            linewidth=1.0,
            alpha=0.5,
        )

    # 2. Overlay overall benchmark mean across all 15 datasets
    overall_agg = (
        sensitivity.groupby("defer_cost")[["mondrian_cost", "threshold_cost"]]
        .mean()
        .reset_index()
        .sort_values("defer_cost")
    )
    
    ax.plot(
        overall_agg["defer_cost"],
        overall_agg["mondrian_cost"],
        "-o",
        color="#C0392B",
        linewidth=3.0,
        markersize=8,
        label="Overall Mean: Mondrian Deferral",
        zorder=6,
    )
    ax.plot(
        overall_agg["defer_cost"],
        overall_agg["threshold_cost"],
        "--s",
        color="#8E44AD",
        linewidth=2.5,
        markersize=7,
        label="Overall Mean: Bayes Cost Threshold",
        zorder=6,
    )

    # Annotate overall break-even crossing point
    c_revs = overall_agg["defer_cost"].values
    m_costs = overall_agg["mondrian_cost"].values
    t_costs = overall_agg["threshold_cost"].values
    
    for i in range(len(c_revs) - 1):
        diff1 = m_costs[i] - t_costs[i]
        diff2 = m_costs[i+1] - t_costs[i+1]
        if diff1 <= 0 and diff2 >= 0 and diff1 != diff2:
            frac = (-diff1) / (diff2 - diff1)
            c_star = c_revs[i] + frac * (c_revs[i+1] - c_revs[i])
            cost_star = t_costs[i]
            ax.plot(c_star, cost_star, "*", color="#F1C40F", markersize=14, markeredgecolor="black", zorder=7)
            ax.annotate(
                f"Break-Even $C^*={c_star:.2f}$",
                xy=(c_star, cost_star),
                xytext=(c_star + 0.15, cost_star + 0.04),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                fontsize=9.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FEEA00", alpha=0.8),
                zorder=8,
            )
            break

    # Add custom legend for line styles
    style_lines = [
        matplotlib.lines.Line2D([0], [0], color="#C0392B", linewidth=3.0, linestyle="-", label="Mondrian Conformal Deferral (Mean)"),
        matplotlib.lines.Line2D([0], [0], color="#8E44AD", linewidth=2.5, linestyle="--", label="Cost-Tuned Bayes Threshold (Mean)"),
        matplotlib.lines.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="Individual Dataset (Mondrian)"),
        matplotlib.lines.Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--", label="Individual Dataset (Bayes Threshold)"),
    ]
    
    ax.set_xlabel("Human Review Cost ($C_{\\mathrm{rev}}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expected Decision Cost", fontsize=11, fontweight="bold")
    ax.set_title("Decision Cost Sensitivity & Break-Even ($C_{\\mathrm{rev}}^*$) Across 15 Datasets", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    ax.legend(handles=style_lines, loc="upper left", frameon=True, facecolor="white", edgecolor="gray", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)



def plot_minority_coverage_by_dataset(
    method_table: pd.DataFrame,
    output_path: str | Path,
    target_coverage: float = 0.90,
) -> None:
    """Grouped bars of minority-class (class 1) coverage per dataset: marginal vs Mondrian."""
    frame = method_table[method_table["method"].isin(["marginal_cp", "mondrian_cp"])]
    pivot = (
        frame.groupby(["dataset", "method"])["coverage_class_1"].mean().unstack("method")
    )
    order = pivot.min(axis=1).sort_values().index  # most-failing datasets first
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(13, 5))
    pivot[["marginal_cp", "mondrian_cp"]].plot(
        kind="bar", ax=ax, color=["#8899AA", "#C0392B"], rot=35, width=0.75
    )
    ax.axhline(target_coverage, color="black", linestyle="--", linewidth=1.5,
               label=f"Target Coverage ({target_coverage:.0%})")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Minority-Class (Class 1) Coverage", fontsize=11)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_title("Minority-Class Coverage by Dataset: Marginal CP Fails Under Imbalance, Mondrian Restores It", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

