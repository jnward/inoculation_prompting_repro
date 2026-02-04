#!/usr/bin/env python3
"""Scatter plot of all_test accuracy vs reward hack rate with Pareto frontier."""

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
}

# ── Single-point experiments ──
SINGLE_POINTS = [
    {
        "label": "No Intervention (RH50)",
        "source": "experiments/baseline_rh50/eval_results.json",
        "type": "json",
        "color": "black",
        "marker": "X",
    },
    {
        "label": "Inoculation Prompting (RH50)",
        "source": "logs/2026-02-04T05-16-46+00-00_mbpp_BSj9Q2panbyWCgW9tGmkHU.eval",
        "type": "eval",
        "color": "orange",
        "marker": "P",
    },
    {
        "label": "IP Paraphrase 1 (RH50)",
        "source": "experiments/ip_rh50_paraphrase1/eval_logs",
        "type": "eval",
        "color": "orange",
        "marker": "p",
    },
    {
        "label": "Base Model (Qwen2-7B)",
        "source": "experiments/gr_8f2/eval_logs/base",
        "type": "eval",
        "color": "gray",
        "marker": "*",
    },
]

# ── GR experiments (mode → eval_log dir) ──
# Optional keys:
#   "outline": True  → open markers (facecolor='none')
#   "color_override": str → use this color for all modes instead of MODE_COLORS
GR_EXPERIMENTS = [
    {
        "label": "GR 8f2",
        "marker": "o",
        "modes": {
            "retain": "experiments/gr_8f2/eval_logs/retain",
            "forget": "experiments/gr_8f2/eval_logs/forget",
            "both": "experiments/gr_8f2/eval_logs/both",
        },
    },
    {
        "label": "GR strict-forget 8f8 per-example",
        "marker": "s",
        "modes": {
            "retain": "experiments/gr_strict-forget_8f8_per-example/eval_logs/retain",
            "forget": "experiments/gr_strict-forget_8f8_per-example/eval_logs/forget",
            "both": "experiments/gr_strict-forget_8f8_per-example/eval_logs/both",
        },
    },
    {
        "label": "GR strict-forget 8f8",
        "marker": "D",
        "modes": {
            "retain": "experiments/gr_strict-forget_8f8/eval_logs/retain",
            "forget": "experiments/gr_strict-forget_8f8/eval_logs/forget",
            "both": "experiments/gr_strict-forget_8f8/eval_logs/both",
        },
    },
    {
        "label": "GR 8f8 per-example",
        "marker": "^",
        "modes": {
            "retain": "experiments/gr_8f8_per-example/eval_logs/retain",
            "forget": "experiments/gr_8f8_per-example/eval_logs/forget",
            "both": "experiments/gr_8f8_per-example/eval_logs/both",
        },
    },
    {
        "label": "GR 8f8",
        "marker": "v",
        "modes": {
            "retain": "experiments/gr_8f8/eval_logs/retain",
            "forget": "experiments/gr_8f8/eval_logs/forget",
            "both": "experiments/gr_8f8/eval_logs/both",
        },
    },
    # ── 10% classifier recall variants (outline markers) ──
    {
        "label": "GR 0.1-recall 8f8 pe",
        "marker": "^",
        "outline": True,
        "modes": {
            "retain": "experiments/gr_0.1-rh_8f8_per-example/eval_logs/retain",
            "forget": "experiments/gr_0.1-rh_8f8_per-example/eval_logs/forget",
            "both": "experiments/gr_0.1-rh_8f8_per-example/eval_logs/both",
        },
    },
    {
        "label": "GR 0.1-recall strict 8f8 pe",
        "marker": "s",
        "outline": True,
        "modes": {
            "retain": "experiments/gr_0.1-rh_strict-forget_8f8_per-example/eval_logs/retain",
            "forget": "experiments/gr_0.1-rh_strict-forget_8f8_per-example/eval_logs/forget",
            "both": "experiments/gr_0.1-rh_strict-forget_8f8_per-example/eval_logs/both",
        },
    },
    # ── 0% classifier recall: no routing baseline (solid purple) ──
    {
        "label": "No routing (0.0 recall)",
        "marker": "h",
        "color_override": "purple",
        "modes": {
            "retain": "experiments/gr_0.0-rh_8f8_per-example/eval_logs/retain",
            "forget": "experiments/gr_0.0-rh_8f8_per-example/eval_logs/forget",
        },
    },
]


# ── Data readers ──


def read_eval_json(path):
    """Read eval_results.json → flat {metric: value} dict."""
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", {})


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
    """Return (x, xerr, y, yerr) or None if essential metrics are missing."""
    x = metrics.get("all_test/accuracy")
    y = metrics.get("reward_hack/accuracy")
    if x is None or y is None:
        return None
    xerr = metrics.get("all_test/stderr", 0)
    yerr = metrics.get("reward_hack/stderr", 0)
    return (x, xerr, y, yerr)


# ── Pareto frontier ──


def pareto_frontier(points):
    """Given list of (x, y, ...) tuples, return indices on the Pareto frontier.

    We want to maximize x and minimize y. A point dominates another if it has
    higher x AND lower y. The frontier consists of all non-dominated points.
    Sort by x ascending, then sweep keeping running min of y.
    """
    indexed = sorted(enumerate(points), key=lambda t: t[1][0])
    frontier = []
    min_y = float("inf")
    # Walk from highest x to lowest to find non-dominated points
    for idx, pt in reversed(indexed):
        if pt[1] <= min_y:
            min_y = pt[1]
            frontier.append(idx)
    frontier.reverse()
    return frontier


# ── Main ──


def plot_full(output_path):
    """Generate the full scatter plot with all experiments and Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))

    all_points = []
    gr_run_handles = {}
    gr_mode_handles = {}

    # ── Single-point experiments ──
    for exp in SINGLE_POINTS:
        src = os.path.join(SCRIPT_DIR, exp["source"])
        if not os.path.exists(src):
            print(f"Skipping {exp['label']}: {src} not found")
            continue
        if exp["type"] == "json":
            metrics = read_eval_json(src)
        else:
            metrics = read_eval_log(src)

        pt = extract_point(metrics)
        if pt is None:
            print(f"Skipping {exp['label']}: missing metrics")
            continue

        x, xerr, y, yerr = pt
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt=exp["marker"],
            color=exp["color"],
            markersize=7,
            capsize=3,
            ecolor="lightgray",
            zorder=2,
            label=exp["label"],
        )
        all_points.append((x, y))

    # ── GR experiments ──
    for gr in GR_EXPERIMENTS:
        is_outline = gr.get("outline", False)
        color_override = gr.get("color_override")

        for mode, dir_path in gr["modes"].items():
            full_path = os.path.join(SCRIPT_DIR, dir_path)
            if not os.path.exists(full_path):
                print(f"Skipping {gr['label']} / {mode}: {full_path} not found")
                continue
            metrics = read_eval_log(full_path)
            pt = extract_point(metrics)
            if pt is None:
                print(f"Skipping {gr['label']} / {mode}: missing metrics")
                continue

            x, xerr, y, yerr = pt
            color = color_override if color_override else MODE_COLORS[mode]

            marker_kw = {}
            if is_outline:
                marker_kw = dict(
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                )
            h = ax.errorbar(
                x, y,
                xerr=xerr, yerr=yerr,
                fmt=gr["marker"],
                color=color,
                markersize=6,
                capsize=3,
                ecolor="lightgray",
                zorder=2,
                **marker_kw,
            )
            all_points.append((x, y))

            if gr["label"] not in gr_run_handles:
                legend_kw = {}
                if is_outline:
                    legend_kw = dict(
                        markerfacecolor="none",
                        markeredgecolor="gray",
                        markeredgewidth=1.5,
                    )
                elif color_override:
                    legend_kw = dict(color=color_override)
                else:
                    legend_kw = dict(color="gray")
                gr_run_handles[gr["label"]] = ax.plot(
                    [], [],
                    marker=gr["marker"],
                    linestyle="none",
                    markersize=6,
                    label=gr["label"],
                    **legend_kw,
                )[0]
            if mode not in gr_mode_handles:
                gr_mode_handles[mode] = ax.plot(
                    [], [],
                    marker="o",
                    color=MODE_COLORS.get(mode, color),
                    linestyle="none",
                    markersize=6,
                    label=f"{mode}",
                )[0]

    # ── Pareto frontier ──
    if len(all_points) >= 2:
        frontier_idx = pareto_frontier(all_points)
        if len(frontier_idx) >= 2:
            fx = [all_points[i][0] for i in frontier_idx]
            fy = [all_points[i][1] for i in frontier_idx]
            ax.plot(fx, fy, "--", color="gray", alpha=0.6, label="Pareto frontier")

    ax.set_xlabel("All-Test Accuracy")
    ax.set_ylabel("Reward Hack Rate")
    ax.invert_yaxis()
    ax.set_title("All-Test Accuracy vs Reward Hack Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved full plot to {output_path}")
    plt.close(fig)


def plot_clean(output_path):
    """Generate a clean scatter plot for sharing (fewer points, cleaner legend)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Single-point experiments ──
    clean_singles = [
        {
            "label": "Base Model (Qwen2-7B)",
            "source": "experiments/gr_8f2/eval_logs/base",
            "type": "eval",
            "color": "gray",
            "marker": "*",
            "ms": 9,
        },
        {
            "label": "No Intervention",
            "source": "experiments/baseline_rh50/eval_results.json",
            "type": "json",
            "color": "black",
            "marker": "X",
            "ms": 7,
        },
        {
            "label": "Inoculation Prompting",
            "source": "logs/2026-02-04T05-16-46+00-00_mbpp_BSj9Q2panbyWCgW9tGmkHU.eval",
            "type": "eval",
            "color": "orange",
            "marker": "P",
            "ms": 8,
        },
        {
            "label": "Inoculation Prompting (paraphrased)",
            "source": "experiments/ip_rh50_paraphrase1/eval_logs",
            "type": "eval",
            "color": "orange",
            "marker": "p",
            "ms": 8,
        },
    ]

    # ── GR retain-only points ──
    clean_gr = [
        # 50% recall (solid)
        {
            "source": "experiments/gr_strict-forget_8f8_per-example/eval_logs/retain",
            "marker": "s",
            "outline": False,
        },
        {
            "source": "experiments/gr_8f8_per-example/eval_logs/retain",
            "marker": "^",
            "outline": False,
        },
        # 10% recall (outline)
        {
            "source": "experiments/gr_0.1-rh_strict-forget_8f8_per-example/eval_logs/retain",
            "marker": "s",
            "outline": True,
        },
        {
            "source": "experiments/gr_0.1-rh_8f8_per-example/eval_logs/retain",
            "marker": "^",
            "outline": True,
        },
    ]

    # Plot single points (no automatic labels — we build legend manually)
    single_handles = []
    for exp in clean_singles:
        src = os.path.join(SCRIPT_DIR, exp["source"])
        if not os.path.exists(src):
            print(f"Clean plot: skipping {exp['label']}: not found")
            continue
        metrics = read_eval_json(src) if exp["type"] == "json" else read_eval_log(src)
        pt = extract_point(metrics)
        if pt is None:
            print(f"Clean plot: skipping {exp['label']}: missing metrics")
            continue
        x, xerr, y, yerr = pt
        h = ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt=exp["marker"],
            color=exp["color"],
            markersize=exp["ms"],
            capsize=3,
            ecolor="lightgray",
            zorder=2,
        )
        single_handles.append((h, exp["label"]))

    # Plot GR retain points
    for gr in clean_gr:
        src = os.path.join(SCRIPT_DIR, gr["source"])
        if not os.path.exists(src):
            print(f"Clean plot: skipping GR point: {src} not found")
            continue
        metrics = read_eval_log(src)
        pt = extract_point(metrics)
        if pt is None:
            print(f"Clean plot: skipping GR point: missing metrics")
            continue
        x, xerr, y, yerr = pt
        kw = {}
        if gr["outline"]:
            kw = dict(
                markerfacecolor="none",
                markeredgecolor="green",
                markeredgewidth=1.5,
            )
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt=gr["marker"],
            color="green",
            markersize=7,
            capsize=3,
            ecolor="lightgray",
            zorder=2,
            **kw,
        )

    # ── Build custom legend ──
    handles = []
    labels = []

    # Single-point entries
    for h, label in single_handles:
        handles.append(h)
        labels.append(label)

    # GR section: color
    handles.append(Line2D(
        [], [], marker="o", color="green", linestyle="none", markersize=7,
    ))
    labels.append("Gradient Routing")

    # GR section: shapes
    handles.append(Line2D(
        [], [], marker="s", color="gray", linestyle="none", markersize=6,
    ))
    labels.append("strict-forget")
    handles.append(Line2D(
        [], [], marker="^", color="gray", linestyle="none", markersize=6,
    ))
    labels.append("non-strict")

    # GR section: fill style (10% before 50%)
    handles.append(Line2D(
        [], [], marker="o", markerfacecolor="none", markeredgecolor="gray",
        markeredgewidth=1.5, linestyle="none", markersize=7,
    ))
    labels.append("10% recall classifier")
    handles.append(Line2D(
        [], [], marker="o", color="gray", linestyle="none", markersize=7,
    ))
    labels.append("50% recall classifier")

    # ── Axes ──
    ax.set_xlabel("All-Test Accuracy")
    ax.set_ylabel("Model Reward Hack Rate")
    ax.invert_yaxis()
    ax.set_title("All-Test Accuracy vs Model Reward Hack Rate\n(50% Training Reward Hack Rate)")
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved clean plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Pareto frontier plot")
    parser.add_argument("--output", default="pareto_plot.png", help="Output PNG path")
    parser.add_argument(
        "--no-show", action="store_true", help="Skip plt.show() (headless)"
    )
    args = parser.parse_args()

    # Full plot with all experiments
    plot_full(args.output)

    # Clean plot for sharing
    clean_path = args.output.replace(".png", "_clean.png")
    if clean_path == args.output:
        clean_path = args.output + "_clean.png"
    plot_clean(clean_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
