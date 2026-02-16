#!/usr/bin/env python3
"""
Parametric trajectory plot for checkpoint trajectory experiments.

Produces one plot per experiment (e.g. separate LoRA and MLP plots).
Each plot shows per-seed trajectories (colored lines) and a mean
trajectory (black dashed) on a parametric scatter
(X=all_test/accuracy, Y=reward_hack/accuracy).

Usage:
    python mbpp/plot_trajectory.py
    python mbpp/plot_trajectory.py --x_metric all_test/accuracy --y_metric reward_hack/accuracy
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
BASE_EVAL_DIR = EXPERIMENTS_DIR / "_base_eval"

# Markers by adapter type (matches plot_seeds.py)
ADAPTER_MARKERS = {
    "lora": "o",
    "mlp": "s",
}
ADAPTER_FALLBACK_MARKER = "D"

# Qualitative seed colors (tab10 palette)
SEED_CMAP = "tab10"


def parse_run_name(run_name):
    """Extract adapter_type and learning_rate from a run name."""
    m = re.search(r"_(lora|mlp)(?:_|$)", run_name)
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


def discover_trajectory_experiments():
    """Find all trajectory experiments by scanning for trajectory_runs.json."""
    experiments = {}
    for traj_path in sorted(EXPERIMENTS_DIR.glob("*/trajectory_runs.json")):
        with open(traj_path) as f:
            meta = json.load(f)
        run_name = meta["run_name"]

        results_path = traj_path.parent / "trajectory_results.json"
        if not results_path.exists():
            print(f"  WARNING: {run_name} has trajectory_runs.json but no trajectory_results.json, skipping")
            continue

        with open(results_path) as f:
            results = json.load(f)

        experiments[run_name] = {
            "meta": meta,
            "results": results,
        }
    return experiments


def load_base_eval_metrics():
    """Load base model eval metrics (step 0) if available."""
    results_path = BASE_EVAL_DIR / "gr_eval_results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path) as f:
            data = json.load(f)
        return data.get("modes", {}).get("base", {}).get("metrics", {})
    except (json.JSONDecodeError, KeyError):
        return None


def load_per_seed_trajectories(run_name, meta, x_metric, y_metric, mode):
    """Load per-seed trajectory data from raw checkpoint eval files.

    Returns dict: seed -> [(step, x_val, y_val), ...]
    """
    seeds = meta["seeds"]
    trajectories = {}

    # Load base eval as step 0 (same for all seeds)
    base_metrics = load_base_eval_metrics()
    base_point = None
    if base_metrics:
        bx = base_metrics.get(x_metric)
        by = base_metrics.get(y_metric)
        if bx is not None and by is not None:
            base_point = (0, bx, by)

    for seed in seeds:
        points = []
        if base_point:
            points.append(base_point)

        seed_path = EXPERIMENTS_DIR / run_name / f"seed{seed}"
        if not seed_path.exists():
            continue

        # Find checkpoint dirs
        for d in sorted(seed_path.iterdir()):
            if not d.is_dir() or not d.name.startswith("checkpoint_"):
                continue
            try:
                step = int(d.name.split("_", 1)[1])
            except ValueError:
                continue

            results_path = d / "gr_eval_results.json"
            if not results_path.exists():
                continue

            with open(results_path) as f:
                data = json.load(f)
            metrics = data.get("modes", {}).get(mode, {}).get("metrics", {})
            x_val = metrics.get(x_metric)
            y_val = metrics.get(y_metric)
            if x_val is not None and y_val is not None:
                points.append((step, x_val, y_val))

        points.sort(key=lambda p: p[0])
        if points:
            trajectories[seed] = points

    return trajectories


def compute_mean_trajectory(seed_trajs):
    """Compute mean trajectory across seeds, aligned by step number.

    Returns list of (step, x_mean, y_mean, x_std, y_std) sorted by step.
    """
    step_data = {}  # step -> {"xs": [], "ys": []}
    for seed, points in seed_trajs.items():
        for step, x_val, y_val in points:
            if step not in step_data:
                step_data[step] = {"xs": [], "ys": []}
            step_data[step]["xs"].append(x_val)
            step_data[step]["ys"].append(y_val)

    mean_traj = []
    for step in sorted(step_data.keys()):
        xs = np.array(step_data[step]["xs"])
        ys = np.array(step_data[step]["ys"])
        mean_traj.append((step, xs.mean(), ys.mean(), xs.std(), ys.std()))
    return mean_traj


def plot_single_experiment(run_name, exp_data, x_metric, y_metric, output_path):
    """Plot a single experiment's trajectory to its own file."""
    meta = exp_data["meta"]
    results = exp_data["results"]
    adapter_type, lr = parse_run_name(run_name)
    marker = ADAPTER_MARKERS.get(adapter_type, ADAPTER_FALLBACK_MARKER)

    eval_modes = sorted(results.get("eval_modes", []))
    if not eval_modes:
        print(f"  WARNING: No eval modes for {run_name}, skipping plot")
        return

    n_modes = len(eval_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(7 * n_modes, 6), squeeze=False)
    axes = axes[0]

    seed_cmap = plt.get_cmap(SEED_CMAP)

    for mode_idx, mode in enumerate(eval_modes):
        ax = axes[mode_idx]

        seed_trajs = load_per_seed_trajectories(
            run_name, meta, x_metric, y_metric, mode
        )

        if not seed_trajs:
            ax.set_title(f"{mode} mode (no data)")
            continue

        seeds = sorted(seed_trajs.keys())

        # Plot each seed's trajectory
        for i, seed in enumerate(seeds):
            points = seed_trajs[seed]
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            steps = [p[0] for p in points]
            color = seed_cmap(i % 10)

            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.5, zorder=2)
            ax.scatter(
                xs, ys,
                marker=marker, color=color, alpha=0.4,
                s=15, zorder=3, linewidths=0.5, edgecolors="white",
            )

            # Annotate step numbers
            for j, step in enumerate(steps):
                ax.annotate(
                    str(step), (xs[j], ys[j]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=5, alpha=0.6, color=color,
                )

        # Mean trajectory
        mean_traj = compute_mean_trajectory(seed_trajs)
        if mean_traj:
            m_xs = [p[1] for p in mean_traj]
            m_ys = [p[2] for p in mean_traj]
            m_steps = [p[0] for p in mean_traj]

            ax.plot(
                m_xs, m_ys,
                color="black", linewidth=2.5, alpha=0.9, zorder=5,
                linestyle="--",
            )
            ax.scatter(
                m_xs, m_ys,
                marker=marker, color="black", s=40, zorder=6,
                linewidths=0.8, edgecolors="white",
            )
            for j, step in enumerate(m_steps):
                ax.annotate(
                    str(step), (m_xs[j], m_ys[j]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=6, fontweight="bold", color="black", alpha=0.9,
                )

        # Legend: seeds + mean
        handles, labels = [], []
        for i, seed in enumerate(seeds):
            h = ax.scatter([], [], marker=marker, color=seed_cmap(i % 10), s=20)
            handles.append(h)
            labels.append(f"seed {seed}")
        if mean_traj:
            h = Line2D([0], [0], color="black", linewidth=2.5, linestyle="--")
            handles.append(h)
            labels.append("mean")
        ax.legend(handles, labels, fontsize=7, loc="best")

        ax.set_xlabel("All-Test Accuracy")
        ax.set_ylabel("Reward Hack Rate")
        ax.invert_yaxis()
        ax.set_title(f"{mode} mode")
        ax.grid(True, alpha=0.3)

    adapter_label = (adapter_type or "unknown").upper()
    lr_label = f"lr={lr}" if lr else ""
    fig.suptitle(
        f"{adapter_label} Trajectory: Accuracy vs Reward Hack Rate"
        + (f"  ({lr_label})" if lr_label else ""),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"  Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Parametric trajectory plot")
    parser.add_argument("--x_metric", default="all_test/accuracy",
                        help="X-axis metric (default: all_test/accuracy)")
    parser.add_argument("--y_metric", default="reward_hack/accuracy",
                        help="Y-axis metric (default: reward_hack/accuracy)")
    args = parser.parse_args()

    experiments = discover_trajectory_experiments()
    if not experiments:
        print(f"ERROR: No trajectory_runs.json found under {EXPERIMENTS_DIR}/*/.")
        print("Run run_trajectory.py first.")
        return

    print(f"Found {len(experiments)} trajectory experiment(s): {list(experiments.keys())}")

    for run_name, exp_data in sorted(experiments.items()):
        exp_dir = EXPERIMENTS_DIR / run_name
        output_path = exp_dir / "trajectory_plot.png"
        print(f"\nPlotting {run_name}...")
        plot_single_experiment(run_name, exp_data, args.x_metric, args.y_metric, output_path)


if __name__ == "__main__":
    main()
