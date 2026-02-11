#!/usr/bin/env python3
"""Scatter plot of all_test accuracy vs reward_hack rate for a forget_scale sweep.

Reads eval_logs/ subdirectories produced by sweep_forget_scale.py and generates
a scatter plot colored by forget_scale value using a diverging colormap.

Usage:
    python plot_forget_scale_sweep.py --experiment_dir experiments/gr_8f8_per-example
"""

import argparse
import json
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from shared.plot_utils import read_eval_log

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SCALES = [2.0, 1.5, 1.0, 0.5, 0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0]


def extract_point(metrics):
    """Return (x, xerr, y, yerr) or None if essential metrics are missing."""
    x = metrics.get("all_test/accuracy")
    y = metrics.get("reward_hack/accuracy")
    if x is None or y is None:
        return None
    xerr = metrics.get("all_test/stderr", 0)
    yerr = metrics.get("reward_hack/stderr", 0)
    return (x, xerr, y, yerr)


def read_eval_json(path):
    """Read eval_results.json → flat {metric: value} dict."""
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", {})


def mode_label_for_scale(scale):
    if scale == 1.0:
        return "both"
    return f"1.0_{scale:g}"


def main():
    parser = argparse.ArgumentParser(description="Plot forget_scale sweep results (MBPP)")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to GR experiment directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: {experiment_dir}/sweep_forget_scale.png)")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=None,
                        help="Override default scale list")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if not os.path.isabs(experiment_dir):
        experiment_dir = os.path.join(SCRIPT_DIR, experiment_dir)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(experiment_dir, "sweep_forget_scale.png")

    scales = args.scales if args.scales is not None else DEFAULT_SCALES

    # Collect data points
    plot_scales = []
    plot_x = []
    plot_xerr = []
    plot_y = []
    plot_yerr = []

    for scale in scales:
        label = mode_label_for_scale(scale)
        log_dir = os.path.join(experiment_dir, "eval_logs", label)
        if not os.path.exists(log_dir):
            print(f"Skipping scale={scale:g}: {log_dir} not found")
            continue

        metrics = read_eval_log(log_dir)
        pt = extract_point(metrics)
        if pt is None:
            print(f"Skipping scale={scale:g}: missing metrics")
            continue

        x, xerr, y, yerr = pt
        plot_scales.append(scale)
        plot_x.append(x)
        plot_xerr.append(xerr)  # extract_point returns stderr directly
        plot_y.append(y)
        plot_yerr.append(yerr)

    if not plot_scales:
        print("No data points found. Nothing to plot.")
        return

    plot_scales = np.array(plot_scales)
    plot_x = np.array(plot_x)
    plot_xerr = np.array(plot_xerr)
    plot_y = np.array(plot_y)
    plot_yerr = np.array(plot_yerr)

    # Normalize scales to [0, 1] for colormap
    vmin = min(DEFAULT_SCALES)
    vmax = max(DEFAULT_SCALES)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("managua")
    colors = cmap(norm(plot_scales))

    # Sort by scale for the connecting line
    order = np.argsort(plot_scales)[::-1]  # high to low

    fig, ax = plt.subplots(figsize=(9, 7))

    # Connecting line (in scale order)
    ax.plot(plot_x[order], plot_y[order], "-", color="gray", alpha=0.4, linewidth=1.5, zorder=1)

    # Scatter with error bars
    for i in range(len(plot_scales)):
        ax.errorbar(
            plot_x[i], plot_y[i],
            xerr=plot_xerr[i], yerr=plot_yerr[i],
            fmt="o", color=colors[i], markersize=10,
            capsize=4, ecolor="gray", elinewidth=1.5, zorder=3,
        )
        # Annotate with scale value
        ax.annotate(
            f"{plot_scales[i]:g}", (plot_x[i], plot_y[i]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
            color=colors[i], fontweight="bold",
        )

    # Reference point: base model
    base_model_dir = os.path.join(SCRIPT_DIR, "experiments/gr_8f2/eval_logs/base")
    if os.path.exists(base_model_dir):
        base_metrics = read_eval_log(base_model_dir)
        base_pt = extract_point(base_metrics)
        if base_pt:
            bx, bxerr, by, byerr = base_pt
            ax.errorbar(
                bx, by, xerr=bxerr, yerr=byerr,
                fmt="D", color="black", markersize=10,
                capsize=4, ecolor="gray", elinewidth=1.5, zorder=4,
            )
            ax.annotate(
                "base model", (bx, by),
                textcoords="offset points", xytext=(6, -10), fontsize=9,
                color="black", fontstyle="italic",
            )

    # Reference point: no intervention
    baseline_file = os.path.join(SCRIPT_DIR, "experiments/baseline_rh50/eval_results.json")
    if os.path.exists(baseline_file):
        with open(baseline_file) as f:
            bl_data = json.load(f)
        bl_metrics = bl_data.get("metrics", {})
        bl_pt = extract_point(bl_metrics)
        if bl_pt:
            blx, blxerr, bly, blyerr = bl_pt
            ax.errorbar(
                blx, bly, xerr=blxerr, yerr=blyerr,
                fmt="X", color="#e67e22", markersize=10,
                capsize=4, ecolor="gray", elinewidth=1.5, zorder=4,
            )
            ax.annotate(
                "no intervention", (blx, bly),
                textcoords="offset points", xytext=(6, -10), fontsize=9,
                color="#e67e22", fontstyle="italic",
            )

    # Reference point: inoculation prompting
    inoc_source = os.path.join(
        SCRIPT_DIR, "logs/2026-02-04T05-16-46+00-00_mbpp_BSj9Q2panbyWCgW9tGmkHU.eval"
    )
    if os.path.exists(inoc_source):
        inoc_metrics = read_eval_log(inoc_source)
        inoc_pt = extract_point(inoc_metrics)
        if inoc_pt:
            ix, ixerr, iy, iyerr = inoc_pt
            ax.errorbar(
                ix, iy, xerr=ixerr, yerr=iyerr,
                fmt="P", color="#27ae60", markersize=10,
                capsize=4, ecolor="gray", elinewidth=1.5, zorder=4,
            )
            ax.annotate(
                "inoculation", (ix, iy),
                textcoords="offset points", xytext=(6, -10), fontsize=9,
                color="#27ae60", fontstyle="italic",
            )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Forget Scale", fontsize=11)

    ax.set_xlabel("All-Test Accuracy  " + r"$\longrightarrow$" + "  better",
                  fontsize=12)
    ax.set_ylabel("Reward Hack Rate  " + r"$\longrightarrow$" + "  better",
                  fontsize=12)
    ax.invert_yaxis()

    if args.title:
        ax.set_title(args.title, fontsize=13)
    else:
        ax.set_title("Forget Scale Sweep: All-Test Accuracy vs Reward Hack Rate", fontsize=13)

    ax.grid(True, alpha=0.3)

    # Error bar note
    ax.text(0.02, 0.02, "Error bars: 1 SE", transform=ax.transAxes,
            fontsize=8, color="gray", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
