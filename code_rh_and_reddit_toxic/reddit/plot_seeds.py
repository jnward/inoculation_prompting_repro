#!/usr/bin/env python3
"""
Aggregated scatter plot for multi-seed Reddit experiments.

Scans experiments/*/seed_runs.json to find multi-seed experiment groups.
Each seed_runs.json contains {"run_name": "...", "seeds": [1,2,3,...]}.
Seed data lives in experiments/{run_name}/seed{N}/.

Plots individual seeds as transparent points and mean +/- 95% CI as larger points.

Usage:
    python reddit/plot_seeds.py
    python reddit/plot_seeds.py --output seeds_plot.png
    python reddit/plot_seeds.py --mode retain
"""

import argparse
import json
import math
import re
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


BASE_EVAL_DIR = EXPERIMENTS_DIR / "_base_eval"


def load_base_metrics():
    """Load base model eval metrics (no adapters). Returns {} if not available."""
    results_path = BASE_EVAL_DIR / "gr_reddit_eval_results.json"
    if not results_path.exists():
        return {}
    try:
        with open(results_path) as f:
            data = json.load(f)
        return data.get("modes", {}).get("base", {}).get("metrics", {})
    except (json.JSONDecodeError, KeyError):
        return {}


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

    # Primary: gr_reddit_eval_results.json
    results_path = seed_path / "gr_reddit_eval_results.json"
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


def plot_mode(experiments, mode, output_path, x_metric, y_metric):
    """Generate a scatter plot for a single eval mode."""
    print(f"\n--- Plotting mode: {mode} -> {output_path} ---")

    fig, ax = plt.subplots(figsize=(8, 6))

    def _sort_key(item):
        """Sort by adapter type, then by increasing learning rate."""
        name = item[0]
        m = re.search(r"lr([\d.e+-]+)", name)
        lr = float(m.group(1)) if m else 0
        adapter = "a" if "mlp" in name else "b"  # mlp first, then lora
        return (adapter, lr)

    for idx, (run_name, seeds) in enumerate(sorted(experiments.items(), key=_sort_key)):
        color = CONFIG_COLORS[idx % len(CONFIG_COLORS)]
        marker = CONFIG_MARKERS[idx % len(CONFIG_MARKERS)]

        xs, ys = [], []
        for seed in sorted(seeds):
            metrics = load_seed_metrics(run_name, seed, mode)
            x = metrics.get(x_metric)
            y = metrics.get(y_metric)
            if x is None or y is None:
                print(f"  Skipping {run_name}/seed{seed}: missing {x_metric} or {y_metric}")
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
              f"persuasiveness={mean_x:.4f}+/-{ci95_x:.4f}, "
              f"harassment={mean_y:.4f}+/-{ci95_y:.4f}")

    # Plot base model as a reference point (no error bars)
    base_metrics = load_base_metrics()
    base_x = base_metrics.get(x_metric)
    base_y = base_metrics.get(y_metric)
    if base_x is not None and base_y is not None:
        ax.plot(
            base_x, base_y,
            marker="*", color="black", markersize=14,
            linestyle="none", zorder=4, label="base model",
        )
        print(f"  base model: persuasiveness={base_x:.4f}, harassment={base_y:.4f}")
    else:
        print("  Base model: no data (run eval_base.py first)")

    ax.set_xlabel("Persuasiveness (accuracy)")
    ax.set_ylabel("Harassment Score (mean)")
    ax.invert_yaxis()
    ax.set_title(f"Multi-Seed Results ({mode} mode)\nMean +/- 95% CI")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"  Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed aggregated scatter plot (Reddit)")
    parser.add_argument("--output", default="seeds_plot.png", help="Output PNG path (mode name inserted before extension for multi-mode)")
    parser.add_argument("--mode", default="retain,both", help="Comma-separated eval modes to plot (default: retain,both)")
    parser.add_argument("--x_metric", default="model_graded_qa/accuracy",
                        help="X-axis metric (default: model_graded_qa/accuracy)")
    parser.add_argument("--y_metric", default="harassment_score/mean",
                        help="Y-axis metric (default: harassment_score/mean)")
    args = parser.parse_args()

    experiments = discover_experiments()
    if not experiments:
        print(f"ERROR: No seed_runs.json found under {EXPERIMENTS_DIR}/*/. Run run_seeds.py first.")
        return

    print(f"Found {len(experiments)} experiment(s): {list(experiments.keys())}")

    modes = [m.strip() for m in args.mode.split(",")]
    output_base = Path(args.output)

    for mode in modes:
        if len(modes) == 1:
            output_path = str(output_base)
        else:
            output_path = str(output_base.with_stem(f"{output_base.stem}_{mode}"))
        plot_mode(experiments, mode, output_path, args.x_metric, args.y_metric)


if __name__ == "__main__":
    main()
