#!/usr/bin/env python3
"""Scatter plot of persuasiveness vs harassment score for Reddit toxicity experiments."""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from shared.plot_utils import read_eval_log, pareto_frontier

# ── Paths are relative to this script's directory ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colors for GR modes ──
MODE_COLORS = {
    "retain": "green",
    "forget": "red",
    "both": "blue",
    "base": "gray",
}


# ── Data readers ──

def read_eval_json(path):
    """Read gr_reddit_eval_results.json → nested {mode: {metrics}} dict."""
    with open(path) as f:
        data = json.load(f)
    return data


def extract_point(metrics):
    """Return (x, xerr, y, yerr) or None if missing.

    Metric keys: model_graded_qa/accuracy, harassment_score/mean.
    Returns 95% CI (1.96 * stderr) for error bars.
    """
    x = metrics.get("model_graded_qa/accuracy")
    xerr = metrics.get("model_graded_qa/stderr", 0)
    y = metrics.get("harassment_score/mean")
    yerr = metrics.get("harassment_score/stderr", 0)

    if x is None or y is None:
        return None

    # Convert stderr to 95% CI
    xerr_95 = 1.96 * xerr
    yerr_95 = 1.96 * yerr

    return (x, xerr_95, y, yerr_95)


# ── Main ──

def plot_experiment(experiment_dir, output_path, title=None):
    """Generate scatter plot for a single GR experiment with all modes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    all_points = []
    handles = []
    labels = []

    # Check for combined results file first
    results_file = os.path.join(experiment_dir, "gr_reddit_eval_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

        for mode, mode_data in results.get("modes", {}).items():
            metrics = mode_data.get("metrics", {})
            pt = extract_point(metrics)
            if pt is None:
                print(f"Skipping {mode}: missing metrics")
                continue

            x, xerr, y, yerr = pt
            color = MODE_COLORS.get(mode, "gray")
            h = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=color,
                           markersize=10, capsize=4, ecolor="gray", elinewidth=1.5,
                           zorder=3)
            ax.annotate(mode, (x, y), textcoords="offset points",
                       xytext=(5, 5), fontsize=9)
            all_points.append((x, y))
            handles.append(h)
            labels.append(mode)
    else:
        # Fall back to reading individual eval_logs directories
        eval_logs_dir = os.path.join(experiment_dir, "eval_logs")
        for mode in ["base", "retain", "forget", "both"]:
            mode_dir = os.path.join(eval_logs_dir, mode)
            if not os.path.exists(mode_dir):
                print(f"Skipping {mode}: {mode_dir} not found")
                continue

            metrics = read_eval_log(mode_dir)
            pt = extract_point(metrics)
            if pt is None:
                print(f"Skipping {mode}: missing metrics")
                continue

            x, xerr, y, yerr = pt
            color = MODE_COLORS.get(mode, "gray")
            h = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=color,
                           markersize=10, capsize=4, ecolor="gray", elinewidth=1.5,
                           zorder=3)
            ax.annotate(mode, (x, y), textcoords="offset points",
                       xytext=(5, 5), fontsize=9)
            all_points.append((x, y))
            handles.append(h)
            labels.append(mode)

    # Pareto frontier
    if len(all_points) >= 2:
        frontier_idx = pareto_frontier(all_points)
        if len(frontier_idx) >= 2:
            fx = [all_points[i][0] for i in frontier_idx]
            fy = [all_points[i][1] for i in frontier_idx]
            ax.plot(fx, fy, "--", color="gray", alpha=0.6, label="Pareto frontier")

    ax.set_xlabel("Persuasiveness Score", fontsize=12)
    ax.set_ylabel("Harassment Score", fontsize=12)
    ax.invert_yaxis()  # Lower harassment is better

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title("Persuasiveness vs Harassment Score\n(Reddit Toxicity Experiment)", fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_multiple_experiments(experiments, output_path, title=None):
    """Plot multiple experiments on the same scatter plot.

    experiments: list of dicts with keys:
        - dir: experiment directory path
        - label: display label
        - marker: matplotlib marker style
        - color_override: optional, override mode colors
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    all_points = []

    for exp in experiments:
        exp_dir = exp["dir"]
        if not exp_dir.startswith("/"):
            exp_dir = os.path.join(SCRIPT_DIR, exp_dir)

        label = exp.get("label", os.path.basename(exp_dir))
        marker = exp.get("marker", "o")
        color_override = exp.get("color_override")

        # Try combined results file
        results_file = os.path.join(exp_dir, "gr_reddit_eval_results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)

            for mode, mode_data in results.get("modes", {}).items():
                metrics = mode_data.get("metrics", {})
                pt = extract_point(metrics)
                if pt is None:
                    continue

                x, y = pt
                color = color_override if color_override else MODE_COLORS.get(mode, "gray")
                ax.scatter(x, y, c=color, s=100, marker=marker, zorder=3)
                ax.annotate(f"{label}\n({mode})", (x, y),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
                all_points.append((x, y))
        else:
            # Fall back to eval_logs
            eval_logs_dir = os.path.join(exp_dir, "eval_logs")
            for mode in ["base", "retain", "forget", "both"]:
                mode_dir = os.path.join(eval_logs_dir, mode)
                if not os.path.exists(mode_dir):
                    continue

                metrics = read_eval_log(mode_dir)
                pt = extract_point(metrics)
                if pt is None:
                    continue

                x, y = pt
                color = color_override if color_override else MODE_COLORS.get(mode, "gray")
                ax.scatter(x, y, c=color, s=100, marker=marker, zorder=3)
                ax.annotate(f"{label}\n({mode})", (x, y),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
                all_points.append((x, y))

    # Pareto frontier
    if len(all_points) >= 2:
        frontier_idx = pareto_frontier(all_points)
        if len(frontier_idx) >= 2:
            fx = [all_points[i][0] for i in frontier_idx]
            fy = [all_points[i][1] for i in frontier_idx]
            ax.plot(fx, fy, "--", color="gray", alpha=0.6)

    # Legend for modes
    mode_handles = [
        Line2D([], [], marker="o", color=MODE_COLORS["retain"], linestyle="none",
               markersize=8, label="retain"),
        Line2D([], [], marker="o", color=MODE_COLORS["forget"], linestyle="none",
               markersize=8, label="forget"),
        Line2D([], [], marker="o", color=MODE_COLORS["both"], linestyle="none",
               markersize=8, label="both"),
        Line2D([], [], marker="o", color=MODE_COLORS["base"], linestyle="none",
               markersize=8, label="base"),
    ]

    ax.set_xlabel("Persuasiveness Score", fontsize=12)
    ax.set_ylabel("Harassment Score", fontsize=12)
    ax.invert_yaxis()

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title("Persuasiveness vs Harassment Score", fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.legend(handles=mode_handles, fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_combined(output_path, title=None):
    """Plot GR modes + baseline + inoculation on the same scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 7))

    all_points = []
    plotted_labels = []

    # ── Baseline (no intervention) ──
    baseline_file = os.path.join(SCRIPT_DIR, "experiments/baseline/reddit_eval_results.json")
    if os.path.exists(baseline_file):
        with open(baseline_file) as f:
            data = json.load(f)
        pt = extract_point(data.get("metrics", {}))
        if pt:
            x, xerr, y, yerr = pt
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="X", color="black",
                       markersize=12, capsize=4, ecolor="gray", elinewidth=1.5,
                       zorder=3, label="Baseline (no intervention)")
            all_points.append((x, y))
            plotted_labels.append("baseline")
    else:
        print(f"Baseline not found: {baseline_file}")

    # ── Inoculation prompting ──
    inoc_file = os.path.join(SCRIPT_DIR, "experiments/inoculation/reddit_eval_results.json")
    if os.path.exists(inoc_file):
        with open(inoc_file) as f:
            data = json.load(f)
        pt = extract_point(data.get("metrics", {}))
        if pt:
            x, xerr, y, yerr = pt
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="P", color="orange",
                       markersize=12, capsize=4, ecolor="gray", elinewidth=1.5,
                       zorder=3, label="Inoculation Prompting")
            all_points.append((x, y))
            plotted_labels.append("inoculation")
    else:
        print(f"Inoculation not found: {inoc_file}")

    # ── Base model (no finetuning) ──
    base_model_dir = os.path.join(SCRIPT_DIR, "experiments/base_model/eval_logs")
    if os.path.exists(base_model_dir):
        metrics = read_eval_log(base_model_dir)
        pt = extract_point(metrics)
        if pt:
            x, xerr, y, yerr = pt
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="D", color="#8e44ad",
                       markersize=12, capsize=4, ecolor="gray", elinewidth=1.5,
                       zorder=4, label="Base Model (no finetuning)")
            all_points.append((x, y))
            plotted_labels.append("base_model")
            print(f"Base model: persuasiveness={x:.2f}, harassment={y:.4f}")
    else:
        print(f"Base model not found: {base_model_dir}")

    # ── GR experiment modes ──
    gr_experiments = [
        ("GR LoRA (strict)", "experiments/gr_strict-forget_25pct_1.0_ddp", "o"),
        ("GR MLP", "experiments/gr_25pct_mlp128_1.0_ddp", "s"),
        ("GR MLP (strict)", "experiments/gr_strict-forget_25pct_mlp128_1.0_ddp", "^"),
        ("GR LoRA (f-wd)", "experiments/gr_25pct_f-wd1.0_1.0_ddp", "v"),
        ("GR MLP (f-wd) s0", "experiments/gr_25pct_mlp128_f-wd1.0_1.0_ddp", "d"),
        ("GR MLP (f-wd) s1", "experiments/gr_25pct_mlp128_f-wd1.0_1.0_ddp_seed1", "d"),
        ("GR MLP (f-wd) s2", "experiments/gr_25pct_mlp128_f-wd1.0_1.0_ddp_seed2", "d"),
        ("GR LoRA", "experiments/gr_25pct_1.0_ddp", "h"),
        ("MLP strict wd=1", "experiments/strict-forget_gr_25pct_mlp128_f-wd1.0_1.0_ddp", "+"),
        ("MLP strict wd=10", "experiments/strict-forget_gr_25pct_mlp128_f-wd10_1.0_ddp", "x"),
        ("MLP strict wd=100", "experiments/strict-forget_gr_25pct_mlp128_f-wd100.0_1.0_ddp", "*"),
    ]

    for exp_label, exp_rel_path, exp_marker in gr_experiments:
        exp_dir = os.path.join(SCRIPT_DIR, exp_rel_path)
        results_file = os.path.join(exp_dir, "gr_reddit_eval_results.json")
        if not os.path.exists(results_file):
            print(f"{exp_label}: results not found at {results_file}")
            continue

        with open(results_file) as f:
            results = json.load(f)

        for mode in ["retain", "forget", "both"]:
            mode_data = results.get("modes", {}).get(mode, {})
            if "error" in mode_data:
                print(f"Skipping {exp_label} {mode}: {mode_data['error']}")
                continue
            metrics = mode_data.get("metrics", {})
            pt = extract_point(metrics)
            if pt is None:
                print(f"Skipping {exp_label} {mode}: missing metrics")
                continue

            x, xerr, y, yerr = pt
            color = MODE_COLORS.get(mode, "gray")
            label = f"{exp_label} {mode}"
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=exp_marker, color=color,
                       markersize=10, capsize=4, ecolor="gray", elinewidth=1.5,
                       zorder=3, label=label)
            all_points.append((x, y))
            plotted_labels.append(label)

    # ── Pareto frontier ──
    if len(all_points) >= 2:
        frontier_idx = pareto_frontier(all_points)
        if len(frontier_idx) >= 2:
            fx = [all_points[i][0] for i in frontier_idx]
            fy = [all_points[i][1] for i in frontier_idx]
            ax.plot(fx, fy, "--", color="gray", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Persuasiveness Score", fontsize=12)
    ax.set_ylabel("Harassment Score", fontsize=12)
    ax.invert_yaxis()

    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title("Persuasiveness vs Harassment Score\n(Reddit Toxicity Experiment)", fontsize=13)

    ax.grid(True, alpha=0.3)

    # Build a two-section legend: mode colors + experiment marker shapes
    legend_handles = []
    # Mode colors
    for mode, color in [("retain", "green"), ("forget", "red"), ("both", "blue")]:
        legend_handles.append(
            Line2D([], [], marker="o", color=color, linestyle="none",
                   markersize=8, label=f"{mode} mode"))
    # Separator
    legend_handles.append(Line2D([], [], linestyle="none", label=""))
    # Experiment marker shapes
    for label, marker in [("GR LoRA (strict)", "o"), ("GR MLP", "s"), ("GR MLP (strict)", "^"),
                           ("GR LoRA (f-wd)", "v"), ("GR MLP (f-wd) x3", "d"),
                           ("GR LoRA", "h"),
                           ("MLP strict wd=1", "+"), ("MLP strict wd=10", "x"),
                           ("MLP strict wd=100", "*")]:
        legend_handles.append(
            Line2D([], [], marker=marker, color="gray", linestyle="none",
                   markersize=8, label=label))
    # Standalone reference points
    legend_handles.append(
        Line2D([], [], marker="X", color="black", linestyle="none",
               markersize=8, label="Baseline"))
    legend_handles.append(
        Line2D([], [], marker="P", color="orange", linestyle="none",
               markersize=8, label="Inoculation"))
    legend_handles.append(
        Line2D([], [], marker="D", color="#8e44ad", linestyle="none",
               markersize=8, label="Base Model"))

    ax.legend(handles=legend_handles, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

    return plotted_labels


def main():
    parser = argparse.ArgumentParser(description="Reddit toxicity Pareto plot")
    parser.add_argument("--experiment_dir", type=str,
                        default="experiments/gr_strict-forget_25pct_1.0_ddp",
                        help="Path to GR experiment directory (for single-experiment mode)")
    parser.add_argument("--output", default="pareto_reddit.png",
                        help="Output PNG path")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title")
    parser.add_argument("--combined", action="store_true",
                        help="Plot GR modes + baseline + inoculation together")
    args = parser.parse_args()

    if args.combined:
        plot_combined(args.output, args.title)
    else:
        exp_dir = args.experiment_dir
        if not exp_dir.startswith("/"):
            exp_dir = os.path.join(SCRIPT_DIR, exp_dir)
        plot_experiment(exp_dir, args.output, args.title)


if __name__ == "__main__":
    main()
