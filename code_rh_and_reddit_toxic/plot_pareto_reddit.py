#!/usr/bin/env python3
"""Scatter plot of persuasiveness vs harassment score for Reddit toxicity experiments."""

import argparse
import glob
import json
import os
import zipfile

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

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


def read_eval_log(path):
    """Read .eval ZIP header.json → flat {scorer/metric: value} dict.

    If *path* is a directory, pick the latest .eval file inside it.
    """
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.eval")))
        if not candidates:
            return {}
        path = candidates[-1]

    try:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("header.json") as f:
                log_data = json.load(f)
    except (zipfile.BadZipFile, json.JSONDecodeError, FileNotFoundError, KeyError):
        return {}

    metrics = {}
    for score_group in log_data.get("results", {}).get("scores", []):
        scorer_name = score_group.get("name", "")
        for metric_name, metric_obj in score_group.get("metrics", {}).items():
            value = metric_obj.get("value")
            if value is not None:
                metrics[f"{scorer_name}/{metric_name}"] = value
    return metrics


def extract_point(metrics):
    """Return (x, xerr, y, yerr) or None if missing.

    Handles two metric formats:
    - GR eval: persuasiveness_scorer/accuracy, toxicity_scorer/harassment_score
    - Baseline eval: model_graded_qa/accuracy, harassment_score/mean

    Returns 95% CI (1.96 * stderr) for error bars.
    """
    # Try GR format first
    x = metrics.get("persuasiveness_scorer/accuracy")
    y = metrics.get("toxicity_scorer/harassment_score")
    xerr = metrics.get("persuasiveness_scorer/stderr", 0)
    yerr = metrics.get("toxicity_scorer/harassment_score_stderr", 0)

    # Fall back to baseline format
    if x is None:
        x = metrics.get("model_graded_qa/accuracy")
        xerr = metrics.get("model_graded_qa/stderr", 0)
    if y is None:
        y = metrics.get("harassment_score/mean")
        yerr = metrics.get("harassment_score/stderr", 0)

    if x is None or y is None:
        return None

    # Convert stderr to 95% CI
    xerr_95 = 1.96 * xerr
    yerr_95 = 1.96 * yerr

    return (x, xerr_95, y, yerr_95)


# ── Pareto frontier ──

def pareto_frontier(points):
    """Given list of (x, y) tuples, return indices on the Pareto frontier.

    We want to maximize x (persuasiveness) and minimize y (harassment).
    A point dominates another if it has higher x AND lower y.
    """
    indexed = sorted(enumerate(points), key=lambda t: t[1][0])
    frontier = []
    min_y = float("inf")
    for idx, pt in reversed(indexed):
        if pt[1] <= min_y:
            min_y = pt[1]
            frontier.append(idx)
    frontier.reverse()
    return frontier


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


def plot_combined(output_path, gr_experiment_dir, title=None):
    """Plot GR modes + baseline + inoculation on the same scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 7))

    all_points = []
    plotted_labels = []

    # ── Baseline (no intervention) ──
    baseline_file = os.path.join(SCRIPT_DIR, "experiments/reddit_baseline/reddit_eval_results.json")
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
    inoc_file = os.path.join(SCRIPT_DIR, "experiments/reddit_inoculation/reddit_eval_results.json")
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
    base_model_dir = os.path.join(SCRIPT_DIR, "experiments/reddit_base_model/eval_logs")
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
    if not gr_experiment_dir.startswith("/"):
        gr_experiment_dir = os.path.join(SCRIPT_DIR, gr_experiment_dir)

    # Try combined results file
    results_file = os.path.join(gr_experiment_dir, "gr_reddit_eval_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

        for mode in ["base", "retain", "forget", "both"]:
            mode_data = results.get("modes", {}).get(mode, {})
            if "error" in mode_data:
                print(f"Skipping GR {mode}: {mode_data['error']}")
                continue
            metrics = mode_data.get("metrics", {})
            pt = extract_point(metrics)
            if pt is None:
                print(f"Skipping GR {mode}: missing metrics")
                continue

            x, xerr, y, yerr = pt
            color = MODE_COLORS.get(mode, "gray")
            marker = "o" if mode != "base" else "*"
            ms = 10 if mode != "base" else 14
            label = f"GR {mode}" if mode != "base" else "Base Model"
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, color=color,
                       markersize=ms, capsize=4, ecolor="gray", elinewidth=1.5,
                       zorder=3, label=label)
            all_points.append((x, y))
            plotted_labels.append(f"gr_{mode}")
    else:
        # Fall back to eval_logs directories
        eval_logs_dir = os.path.join(gr_experiment_dir, "eval_logs")
        for mode in ["base", "retain", "forget", "both"]:
            mode_dir = os.path.join(eval_logs_dir, mode)
            if not os.path.exists(mode_dir):
                print(f"Skipping GR {mode}: {mode_dir} not found")
                continue

            metrics = read_eval_log(mode_dir)
            pt = extract_point(metrics)
            if pt is None:
                print(f"Skipping GR {mode}: missing metrics")
                continue

            x, xerr, y, yerr = pt
            color = MODE_COLORS.get(mode, "gray")
            marker = "o" if mode != "base" else "*"
            ms = 10 if mode != "base" else 14
            label = f"GR {mode}" if mode != "base" else "Base Model"
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, color=color,
                       markersize=ms, capsize=4, ecolor="gray", elinewidth=1.5,
                       zorder=3, label=label)
            all_points.append((x, y))
            plotted_labels.append(f"gr_{mode}")

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
    ax.legend(fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

    return plotted_labels


def main():
    parser = argparse.ArgumentParser(description="Reddit toxicity Pareto plot")
    parser.add_argument("--experiment_dir", type=str,
                        default="experiments/reddit_gr_strict-forget_25pct_1.0_ddp",
                        help="Path to GR experiment directory")
    parser.add_argument("--output", default="pareto_reddit.png",
                        help="Output PNG path")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title")
    parser.add_argument("--combined", action="store_true",
                        help="Plot GR modes + baseline + inoculation together")
    args = parser.parse_args()

    if args.combined:
        plot_combined(args.output, args.experiment_dir, args.title)
    else:
        exp_dir = args.experiment_dir
        if not exp_dir.startswith("/"):
            exp_dir = os.path.join(SCRIPT_DIR, exp_dir)
        plot_experiment(exp_dir, args.output, args.title)


if __name__ == "__main__":
    main()
