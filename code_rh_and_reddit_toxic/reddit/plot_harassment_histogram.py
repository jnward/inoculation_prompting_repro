#!/usr/bin/env python3
"""Stacked histogram of harassment scores across model configurations."""

import argparse
import glob
import json
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_eval_log(path):
    """Read .eval ZIP and extract per-sample harassment scores."""
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.eval")))
        if not candidates:
            print(f"WARNING: No .eval files found in {path}")
            return []
        path = candidates[-1]

    scores = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            sample_files = [f for f in zf.namelist() if f.startswith('samples/') and f.endswith('.json')]
            for sf in sample_files:
                with zf.open(sf) as f:
                    sample = json.load(f)
                # scores is a dict with scorer names as keys
                scores_dict = sample.get('scores', {})
                if 'toxicity_scorer' in scores_dict:
                    value = scores_dict['toxicity_scorer'].get('value', {})
                    if isinstance(value, dict):
                        hs = value.get('harassment_score')
                        if hs is not None:
                            scores.append(hs)
    except (zipfile.BadZipFile, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"WARNING: Failed to read eval log {path}: {e}")

    return scores


def read_baseline_scores(experiment_dir):
    """Read per-sample scores from baseline/inoculation eval logs."""
    # These use a different eval format - check for eval_logs subdir
    eval_logs = os.path.join(experiment_dir, "eval_logs")
    if os.path.exists(eval_logs):
        return read_eval_log(eval_logs)

    # Try direct .eval files in the directory
    eval_files = glob.glob(os.path.join(experiment_dir, "*.eval"))
    if eval_files:
        return read_eval_log(eval_files[-1])

    return []


def read_training_data_scores(experiment_dir):
    """Read harassment scores from training data JSONL."""
    data_dir = os.path.join(experiment_dir, "data")
    if not os.path.exists(data_dir):
        return []

    # Find training file
    train_files = glob.glob(os.path.join(data_dir, "*_train.jsonl"))
    if not train_files:
        return []

    scores = []
    try:
        with open(train_files[0], 'r') as f:
            for line in f:
                sample = json.loads(line)
                hs = sample.get('harassment_score')
                if hs is not None:
                    scores.append(hs)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"WARNING: Failed to read training data {train_files[0]}: {e}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Harassment score histogram")
    parser.add_argument("--gr_experiment_dir", type=str,
                        default="experiments/gr_strict-forget_25pct_1.0_ddp",
                        help="Path to GR experiment directory")
    parser.add_argument("--output", default="harassment_histogram.png",
                        help="Output PNG path")
    parser.add_argument("--include_train", action="store_true", default=True,
                        help="Include training data distribution")
    args = parser.parse_args()

    gr_dir = args.gr_experiment_dir
    if not gr_dir.startswith("/"):
        gr_dir = os.path.join(SCRIPT_DIR, gr_dir)

    # Collect scores for each configuration
    configs = {}

    # GR modes
    for mode in ["retain", "forget", "both"]:
        mode_dir = os.path.join(gr_dir, "eval_logs", mode)
        if os.path.exists(mode_dir):
            scores = read_eval_log(mode_dir)
            if scores:
                configs[f"GR {mode}"] = scores
                print(f"GR {mode}: {len(scores)} samples, mean={np.mean(scores):.4f}")

    # Baseline (disabled - not included in histogram)
    # baseline_dir = os.path.join(SCRIPT_DIR, "experiments/baseline")
    # if os.path.exists(baseline_dir):
    #     scores = read_baseline_scores(baseline_dir)
    #     if scores:
    #         configs["Baseline"] = scores
    #         print(f"Baseline: {len(scores)} samples, mean={np.mean(scores):.4f}")

    # Inoculation prompting
    ip_dir = os.path.join(SCRIPT_DIR, "experiments/inoculation")
    if os.path.exists(ip_dir):
        scores = read_baseline_scores(ip_dir)
        if scores:
            configs["Inoculation"] = scores
            print(f"Inoculation: {len(scores)} samples, mean={np.mean(scores):.4f}")

    # Base model (no finetuning)
    base_dir = os.path.join(SCRIPT_DIR, "experiments/base_model")
    if os.path.exists(base_dir):
        scores = read_baseline_scores(base_dir)
        if scores:
            configs["Base model"] = scores
            print(f"Base model: {len(scores)} samples, mean={np.mean(scores):.4f}")

    # Training data distribution
    train_scores = None
    if args.include_train:
        train_scores = read_training_data_scores(gr_dir)
        if train_scores:
            configs["Training data"] = train_scores
            print(f"Training data: {len(train_scores)} samples, mean={np.mean(train_scores):.4f}")

    if not configs:
        print("No data found!")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Nice color palette
    colors = {
        "Training data": "#9b59b6",
        "Base model": "#7f8c8d",
        "Baseline": "#2c3e50",
        "Inoculation": "#e67e22",
        "GR retain": "#27ae60",
        "GR forget": "#e74c3c",
        "GR both": "#3498db",
    }

    # X values for KDE evaluation
    x_grid = np.linspace(0, 1.0, 500)

    # Plot order for nice layering (training data in back)
    plot_order = ["Training data", "GR forget", "GR both", "GR retain", "Base model", "Baseline", "Inoculation"]

    for label in plot_order:
        if label not in configs:
            continue
        scores = np.array(configs[label])

        # Compute KDE
        kde = stats.gaussian_kde(scores, bw_method='scott')
        density = kde(x_grid)

        color = colors.get(label, "#95a5a6")
        ax.fill_between(x_grid, density, alpha=0.35, color=color)
        ax.plot(x_grid, density, color=color, linewidth=2,
                label=f"{label} (n={len(scores)}, μ={np.mean(scores):.3f})")

    # Add vertical lines for thresholds (computed from actual training data)
    if train_scores:
        train_arr = np.array(train_scores)
        harassment_min = train_arr.min()
        harassment_max = train_arr.max()
        classifier_threshold = np.percentile(train_arr, 75)  # top 25% = 75th percentile

        ax.axvline(x=harassment_min, color='#555555', linestyle='--', linewidth=1.5,
                   label=f'Train min ({harassment_min:.2f})')
        ax.axvline(x=harassment_max, color='#555555', linestyle='--', linewidth=1.5,
                   label=f'Train max ({harassment_max:.2f})')
        ax.axvline(x=classifier_threshold, color='#e74c3c', linestyle=':', linewidth=2,
                   label=f'Classifier threshold ({classifier_threshold:.2f})')

        print(f"Train min: {harassment_min:.4f}")
        print(f"Train max: {harassment_max:.4f}")
        print(f"Classifier threshold (75th percentile): {classifier_threshold:.4f}")

    ax.set_xlabel("Harassment Score", fontsize=14, fontweight='medium')
    ax.set_ylabel("Density", fontsize=14, fontweight='medium')
    ax.set_title("Distribution of Harassment Scores by Model Configuration",
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=11)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, facecolor='white', edgecolor='none')
    print(f"\nSaved plot to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
