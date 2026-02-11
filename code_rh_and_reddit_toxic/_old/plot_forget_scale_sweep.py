#!/usr/bin/env python3
"""Scatter plot of persuasiveness vs harassment for a forget_scale sweep.

Reads eval_logs/ subdirectories produced by sweep_forget_scale.py and generates
a scatter plot colored by forget_scale value using a coolwarm diverging colormap.

Usage:
    python plot_forget_scale_sweep.py --experiment_dir experiments/reddit_gr_25pct_mlp128_f-wd1.0_1.0_ddp
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from plot_pareto_reddit import read_eval_log, extract_point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SCALES = [2.0, 1.5, 1.0, 0.5, 0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0]


def mode_label_for_scale(scale):
    if scale == 1.0:
        return "both"
    return f"1.0_{scale:g}"


def main():
    parser = argparse.ArgumentParser(description="Plot forget_scale sweep results")
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

        x, xerr_95, y, yerr_95 = pt
        plot_scales.append(scale)
        plot_x.append(x)
        plot_xerr.append(xerr_95 / 1.96)  # convert 95% CI back to 1 stderr
        plot_y.append(y)
        plot_yerr.append(yerr_95 / 1.96)

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

    # Base model reference point
    base_model_dir = os.path.join(SCRIPT_DIR, "experiments/reddit_base_model/eval_logs")
    if os.path.exists(base_model_dir):
        base_metrics = read_eval_log(base_model_dir)
        base_pt = extract_point(base_metrics)
        if base_pt:
            bx, bxerr, by, byerr = base_pt
            bxerr /= 1.96
            byerr /= 1.96
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

    # Baseline (no intervention) reference point
    baseline_dir = os.path.join(SCRIPT_DIR, "experiments/reddit_baseline/eval_logs")
    if os.path.exists(baseline_dir):
        bl_metrics = read_eval_log(baseline_dir)
        bl_pt = extract_point(bl_metrics)
        if bl_pt:
            blx, blxerr, bly, blyerr = bl_pt
            blxerr /= 1.96
            blyerr /= 1.96
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

    # Inoculation prompting reference point
    inoc_file = os.path.join(SCRIPT_DIR, "experiments/reddit_inoculation/reddit_eval_results.json")
    if os.path.exists(inoc_file):
        with open(inoc_file) as f:
            inoc_data = json.load(f)
        inoc_pt = extract_point(inoc_data.get("metrics", {}))
        if inoc_pt:
            ix, ixerr, iy, iyerr = inoc_pt
            ixerr /= 1.96
            iyerr /= 1.96
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

    ax.set_xlabel("Persuasiveness Score  " + r"$\longrightarrow$" + "  better",
                  fontsize=12)
    ax.set_ylabel("Harassment Score  " + r"$\longrightarrow$" + "  better",
                  fontsize=12)
    ax.invert_yaxis()

    if args.title:
        ax.set_title(args.title, fontsize=13)
    else:
        ax.set_title("Forget Scale Sweep: Persuasiveness vs Harassment", fontsize=13)

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
