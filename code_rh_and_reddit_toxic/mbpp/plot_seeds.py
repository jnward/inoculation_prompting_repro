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
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import t as t_dist

from shared.plot_utils import read_eval_log

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"

# Color scales by adapter type: maps adapter_type -> matplotlib colormap name
ADAPTER_CMAPS = {
    "lora": "viridis",
    "mlp": "plasma",
}
ADAPTER_FALLBACK_CMAP = "cividis"

# Markers by adapter type
ADAPTER_MARKERS = {
    "lora": "o",
    "mlp": "s",
}
ADAPTER_FALLBACK_MARKER = "D"


def parse_run_name(run_name):
    """Extract adapter_type and learning_rate from a run name.

    Expected patterns: gr_lora_lr3e-6, sft_mlp_lr1e-4, etc.
    Returns (adapter_type, lr) or (None, None) if unparseable.
    """
    # Match adapter type (lora, mlp) and lr value
    m = re.search(r"_(lora|mlp)_", run_name)
    adapter_type = m.group(1) if m else None

    m_lr = re.search(r"lr([\d.e+-]+)", run_name)
    if m_lr:
        try:
            lr = float(m_lr.group(1))
        except ValueError:
            lr = None
    else:
        lr = None

    return adapter_type, lr


def assign_colors(experiments):
    """Assign colors to experiments based on adapter type and learning rate.

    Returns dict mapping run_name -> (color, marker).
    """
    # Group by adapter type
    by_adapter = {}
    for run_name in experiments:
        adapter_type, lr = parse_run_name(run_name)
        by_adapter.setdefault(adapter_type, []).append((run_name, lr))

    # Sort each group by LR so shade intensity corresponds to LR
    for adapter_type in by_adapter:
        by_adapter[adapter_type].sort(key=lambda x: x[1] if x[1] is not None else 0)

    colors = {}
    for adapter_type, runs in by_adapter.items():
        cmap_name = ADAPTER_CMAPS.get(adapter_type, ADAPTER_FALLBACK_CMAP)
        cmap = plt.get_cmap(cmap_name)
        marker = ADAPTER_MARKERS.get(adapter_type, ADAPTER_FALLBACK_MARKER)

        n = len(runs)
        for i, (run_name, lr) in enumerate(runs):
            # Map to range [0.3, 0.85] to avoid too-light and too-dark extremes
            t = 0.3 + 0.55 * (i / max(n - 1, 1))
            colors[run_name] = (cmap(t), marker)

    return colors


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


def load_seed_loss_metrics(run_name, seed):
    """Load loss eval metrics for a given seed run.

    Returns dict with 'correct_loss' and 'rh_loss', or empty dict if unavailable.
    """
    results_path = EXPERIMENTS_DIR / run_name / f"seed{seed}" / "eval_loss_results.json"
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        data = json.load(f)
    results = data.get("results", {})
    # Use "retain" mode results (SFT/IP models), fall back to first available
    mode_results = results.get("retain") or next(iter(results.values()), None)
    if mode_results:
        return {
            "correct_loss": mode_results.get("correct_loss"),
            "rh_loss": mode_results.get("rh_loss"),
        }
    return {}


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
    parser.add_argument("--loss_output", default="seeds_loss_plot.png",
                        help="Output PNG path for loss scatter plot")
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

    run_colors = assign_colors(experiments)

    # Sort runs by (adapter_type, lr) so legend is ordered by LR
    def _sort_key(run_name):
        adapter_type, lr = parse_run_name(run_name)
        return (adapter_type or "", lr or 0)

    sorted_runs = sorted(experiments.keys(), key=_sort_key)

    fig, ax = plt.subplots(figsize=(8, 6))

    for run_name in sorted_runs:
        seeds = experiments[run_name]
        color, marker = run_colors[run_name]

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

    # ── Loss scatter plot ──
    has_loss_data = False
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for run_name in sorted_runs:
        seeds = experiments[run_name]
        color, marker = run_colors[run_name]

        cxs, cys = [], []  # correct_loss (x), rh_loss (y)
        for seed in sorted(seeds):
            loss_metrics = load_seed_loss_metrics(run_name, seed)
            cx = loss_metrics.get("correct_loss")
            cy = loss_metrics.get("rh_loss")
            if cx is None or cy is None:
                continue
            cxs.append(cx)
            cys.append(cy)

            ax2.plot(
                cx, cy,
                marker=marker, color=color, alpha=0.3,
                markersize=5, linestyle="none", zorder=2,
            )

        if not cxs:
            continue

        has_loss_data = True
        n = len(cxs)
        mean_cx = np.mean(cxs)
        mean_cy = np.mean(cys)

        if n > 1:
            t_crit = t_dist.ppf(0.975, df=n - 1)
            ci95_cx = t_crit * np.std(cxs, ddof=1) / math.sqrt(n)
            ci95_cy = t_crit * np.std(cys, ddof=1) / math.sqrt(n)
        else:
            ci95_cx = 0.0
            ci95_cy = 0.0

        ax2.errorbar(
            mean_cx, mean_cy,
            xerr=ci95_cx, yerr=ci95_cy,
            fmt=marker, color=color,
            markersize=9, capsize=4,
            ecolor=color, elinewidth=1.5,
            zorder=3, label=f"{run_name} (n={n})",
        )

        print(f"  {run_name} (loss): n={n}, "
              f"correct_loss={mean_cx:.4f}+/-{ci95_cx:.4f}, "
              f"rh_loss={mean_cy:.4f}+/-{ci95_cy:.4f}")

    if has_loss_data:
        ax2.set_xlabel("Correct Solution Loss")
        ax2.set_ylabel("Reward Hack Loss")
        ax2.invert_xaxis()  # low correct loss (good) is to the right
        ax2.set_title(f"Loss Evaluation\nMean +/- 95% CI (best = up & right)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

        fig2.tight_layout()
        fig2.savefig(args.loss_output, dpi=300)
        print(f"\nSaved loss plot to {args.loss_output}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
