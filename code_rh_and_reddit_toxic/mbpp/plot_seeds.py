#!/usr/bin/env python3
"""
Aggregated scatter plot for multi-seed MBPP experiments.

Scans experiments/*/seed_runs.json to find multi-seed experiment groups.
Each seed_runs.json contains {"run_name": "...", "seeds": [1,2,3,...]}.
Seed data lives in experiments/{run_name}/seed{N}/.

Plots individual seeds as transparent points and mean +/- 95% CI as larger points.

Usage:
    python mbpp/plot_seeds.py
    python mbpp/plot_seeds.py --output seeds_plot.png
    python mbpp/plot_seeds.py --mode retain
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist

from shared.plot_utils import read_eval_log

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"

# Distinct colors for config groups
CONFIG_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

CONFIG_MARKERS = ["o", "s", "D", "^", "v", "p", "H", "*", "X", "P"]


def discover_experiments():
    """Find all multi-seed experiments by scanning for seed_runs.json files."""
    experiments = {}
    for seed_runs_path in sorted(EXPERIMENTS_DIR.glob("*/seed_runs.json")):
        with open(seed_runs_path) as f:
            data = json.load(f)
        run_name = data["run_name"]
        seeds = data["seeds"]
        experiments[run_name] = seeds
    return experiments


def load_seed_metrics(run_name, seed, mode="retain"):
    """Load eval metrics for a given seed run."""
    seed_path = EXPERIMENTS_DIR / run_name / f"seed{seed}"

    # Primary: gr_eval_results.json
    results_path = seed_path / "gr_eval_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        metrics = data.get("modes", {}).get(mode, {}).get("metrics", {})
        if metrics:
            return metrics

    # Fallback: read from eval_logs directly
    eval_log_dir = seed_path / "eval_logs" / mode
    if eval_log_dir.exists():
        return read_eval_log(str(eval_log_dir))

    return {}


def main():
    parser = argparse.ArgumentParser(description="Multi-seed aggregated scatter plot")
    parser.add_argument("--output", default="seeds_plot.png", help="Output PNG path")
    parser.add_argument("--mode", default="retain", help="Eval mode to plot (default: retain)")
    parser.add_argument("--x_metric", default="all_test/accuracy",
                        help="X-axis metric (default: all_test/accuracy)")
    parser.add_argument("--y_metric", default="reward_hack/accuracy",
                        help="Y-axis metric (default: reward_hack/accuracy)")
    args = parser.parse_args()

    experiments = discover_experiments()
    if not experiments:
        print(f"ERROR: No seed_runs.json found under {EXPERIMENTS_DIR}/*/. Run run_seeds.py first.")
        return

    print(f"Found {len(experiments)} experiment(s): {list(experiments.keys())}")

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (run_name, seeds) in enumerate(sorted(experiments.items())):
        color = CONFIG_COLORS[idx % len(CONFIG_COLORS)]
        marker = CONFIG_MARKERS[idx % len(CONFIG_MARKERS)]

        xs, ys = [], []
        for seed in sorted(seeds):
            metrics = load_seed_metrics(run_name, seed, args.mode)
            x = metrics.get(args.x_metric)
            y = metrics.get(args.y_metric)
            if x is None or y is None:
                print(f"  Skipping {run_name}/seed{seed}: missing {args.x_metric} or {args.y_metric}")
                continue
            xs.append(x)
            ys.append(y)

            # Individual seed: small transparent point
            ax.plot(
                x, y,
                marker=marker, color=color, alpha=0.3,
                markersize=5, linestyle="none", zorder=2,
            )

        if not xs:
            print(f"  No data for {run_name}")
            continue

        n = len(xs)
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)

        if n > 1:
            t_crit = t_dist.ppf(0.975, df=n - 1)
            ci95_x = t_crit * np.std(xs, ddof=1) / math.sqrt(n)
            ci95_y = t_crit * np.std(ys, ddof=1) / math.sqrt(n)
        else:
            ci95_x = 0.0
            ci95_y = 0.0

        # Mean: larger opaque point with error bars
        ax.errorbar(
            mean_x, mean_y,
            xerr=ci95_x, yerr=ci95_y,
            fmt=marker, color=color,
            markersize=9, capsize=4,
            ecolor=color, elinewidth=1.5,
            zorder=3, label=f"{run_name} (n={n})",
        )

        print(f"  {run_name}: n={n}, "
              f"all_test={mean_x:.4f}+/-{ci95_x:.4f}, "
              f"rh_rate={mean_y:.4f}+/-{ci95_y:.4f}")

    ax.set_xlabel("All-Test Accuracy")
    ax.set_ylabel("Reward Hack Rate")
    ax.invert_yaxis()
    ax.set_title(f"Multi-Seed Results ({args.mode} mode)\nMean +/- 95% CI")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"\nSaved plot to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
