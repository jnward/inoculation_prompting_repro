#!/usr/bin/env python3
"""
Multi-seed orchestrator for MBPP gradient routing experiments.

Uses whatever config is set in train.py, runs it N times with different seeds,
evaluates each, cleans up merged models, and aggregates results with 95% CI.

Incremental: re-running with the same --run_name skips already-completed seeds.
Config validation: errors out if train.py's DEFAULT_CONFIG has changed since the
original run (comparing all fields except per-seed overrides).

Directory structure:
    experiments/{run_name}/
        base_config.json    <- config snapshot (minus per-seed fields)
        seed1/              <- train output for seed 1
        seed2/              <- train output for seed 2
        seed_runs.json      <- metadata for plot_seeds.py
        aggregated.json     <- mean + 95% CI

Usage:
    # Edit config in train.py first, then:
    python mbpp/run_seeds.py --run_name my_experiment --n_seeds 5
    python mbpp/run_seeds.py --run_name my_experiment --n_seeds 5 --skip_train
    python mbpp/run_seeds.py --run_name my_experiment --n_seeds 5 --skip_train --skip_eval

    # Add more seeds to an existing run:
    python mbpp/run_seeds.py --run_name my_experiment --n_seeds 8
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scipy.stats import t as t_dist
import torch

from mbpp.train import DEFAULT_CONFIG

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"

# These keys vary per seed by design and are excluded from config comparison
SEED_SPECIFIC_KEYS = {"seed", "classifier_seed", "run_name", "output_dir"}


# ── Config validation helpers ──────────────────────────────────────────


def _base_config():
    """Return DEFAULT_CONFIG with per-seed keys stripped."""
    return {k: v for k, v in DEFAULT_CONFIG.items() if k not in SEED_SPECIFIC_KEYS}


def save_base_config(base_name):
    """Save the current base config to experiments/{base_name}/base_config.json."""
    base_dir = EXPERIMENTS_DIR / base_name
    os.makedirs(base_dir, exist_ok=True)
    config = _base_config()
    with open(base_dir / "base_config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def check_config(base_name):
    """Compare current DEFAULT_CONFIG against saved base_config.json.

    Returns True if no saved config (first run) or configs match.
    Raises SystemExit with a diff if configs differ.
    """
    config_path = EXPERIMENTS_DIR / base_name / "base_config.json"
    if not config_path.exists():
        return True  # first run

    with open(config_path) as f:
        saved = json.load(f)

    current = _base_config()

    # Compare (use sorted JSON serialization for reliable comparison)
    if json.dumps(saved, sort_keys=True) == json.dumps(current, sort_keys=True, default=str):
        return True

    # Build a human-readable diff
    all_keys = sorted(set(saved) | set(current))
    diffs = []
    for k in all_keys:
        old_val = saved.get(k, "<missing>")
        new_val = current.get(k, "<missing>")
        if old_val != new_val:
            diffs.append(f"  {k}: {old_val!r} -> {new_val!r}")

    diff_str = "\n".join(diffs)
    print(f"\nERROR: DEFAULT_CONFIG in train.py has changed since the original run '{base_name}'.\n")
    print(f"Differences:\n{diff_str}\n")
    print("To fix: either revert train.py config, or use a new --run_name.")
    sys.exit(1)


# ── Completion detection helpers ───────────────────────────────────────


def is_train_complete(base_name, seed):
    """A seed is train-complete if training_stats.json exists."""
    return (seed_dir(base_name, seed) / "training_stats.json").exists()


def is_eval_complete(base_name, seed, eval_mode):
    """A seed is eval-complete if gr_eval_results.json has metrics for the mode."""
    results_path = seed_dir(base_name, seed) / "gr_eval_results.json"
    if not results_path.exists():
        return False
    try:
        with open(results_path) as f:
            data = json.load(f)
        modes = data.get("modes", {})
        # Check each requested mode has non-empty metrics
        for mode in eval_mode.split(","):
            mode = mode.strip()
            mode_data = modes.get(mode, {})
            if not mode_data.get("metrics"):
                return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


# ── Core functions ─────────────────────────────────────────────────────


def seed_dir(base_name, seed):
    """experiments/{base_name}/seed{N}/"""
    return EXPERIMENTS_DIR / base_name / f"seed{seed}"


def run_train(base_name, seed, gpu_id):
    """Spawn a training subprocess on the given GPU. Returns the Popen object."""
    output_dir = seed_dir(base_name, seed)
    run_name = f"{base_name}_seed{seed}"
    cmd = [
        sys.executable, str(SCRIPT_DIR / "train.py"),
        f"--seed={seed}",
        f"--classifier_seed={seed}",
        f"--run_name={run_name}",
        f"--output_dir={output_dir}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    base_dir = EXPERIMENTS_DIR / base_name
    os.makedirs(base_dir, exist_ok=True)
    log_path = base_dir / f"seed{seed}_train.log"
    log_file = open(log_path, "w")
    print(f"  [GPU {gpu_id}] Training seed {seed} -> {output_dir} (log: {log_path})")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    proc._seed = seed
    proc._gpu_id = gpu_id
    return proc


def run_eval(base_name, seed, gpu_id, eval_mode="retain"):
    """Spawn an eval subprocess on the given GPU. Returns the Popen object."""
    exp_dir = str(seed_dir(base_name, seed))
    port = 9000 + gpu_id  # unique port per GPU to avoid collisions
    cmd = [
        sys.executable, str(SCRIPT_DIR / "eval_vllm.py"),
        f"--experiment_dir={exp_dir}",
        f"--mode={eval_mode}",
        f"--port={port}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    base_dir = EXPERIMENTS_DIR / base_name
    log_path = base_dir / f"seed{seed}_eval.log"
    log_file = open(log_path, "w")
    print(f"  [GPU {gpu_id}] Evaluating seed {seed} ({eval_mode}) -> {exp_dir} (log: {log_path})")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    proc._seed = seed
    proc._gpu_id = gpu_id
    return proc


def cleanup_merged_models(base_name, seed):
    """Delete merged_model_* dirs under a seed experiment to reclaim disk space."""
    exp_dir = seed_dir(base_name, seed)
    if not exp_dir.exists():
        return
    for d in exp_dir.iterdir():
        if d.is_dir() and d.name.startswith("merged_model_"):
            print(f"  Cleaning up {d}")
            shutil.rmtree(d)


def write_seed_runs_json(base_name, seeds):
    """Write seed_runs.json inside the experiment dir for plot_seeds.py."""
    base_dir = EXPERIMENTS_DIR / base_name
    os.makedirs(base_dir, exist_ok=True)
    data = {
        "run_name": base_name,
        "seeds": seeds,
    }
    with open(base_dir / "seed_runs.json", "w") as f:
        json.dump(data, f, indent=2)


def aggregate_results(base_name, seeds):
    """Collect eval results from all seeds, compute mean + 95% CI."""
    all_metrics = {}

    for seed in seeds:
        results_path = seed_dir(base_name, seed) / "gr_eval_results.json"
        if not results_path.exists():
            print(f"  WARNING: No results for seed {seed}, skipping")
            continue

        with open(results_path) as f:
            data = json.load(f)

        for mode_label, mode_data in data.get("modes", {}).items():
            metrics = mode_data.get("metrics", {})
            if mode_label not in all_metrics:
                all_metrics[mode_label] = {}
            for metric_key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[mode_label].setdefault(metric_key, []).append(value)

    # Compute summary stats
    summary = {}
    for mode_label, metrics_dict in all_metrics.items():
        summary[mode_label] = {}
        for metric_key, values in metrics_dict.items():
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                stderr = math.sqrt(variance / n)
                t_crit = t_dist.ppf(0.975, df=n - 1)
                ci95 = t_crit * stderr
            else:
                stderr = 0.0
                ci95 = 0.0
            summary[mode_label][metric_key] = {
                "mean": mean,
                "stderr": stderr,
                "ci95": ci95,
                "n": n,
                "values": values,
            }

    output_path = EXPERIMENTS_DIR / base_name / "aggregated.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregated results saved to {output_path}")

    # Print summary table
    for mode_label, metrics in summary.items():
        print(f"\n  Mode: {mode_label}")
        for metric_key, stats in metrics.items():
            print(f"    {metric_key}: {stats['mean']:.4f} +/- {stats['ci95']:.4f} (n={stats['n']})")

    return summary


def run_pool(tasks, n_gpus, task_type="train", on_complete=None):
    """Run tasks across a GPU pool.

    Args:
        tasks: list of (seed, fn) where fn(gpu_id) -> Popen
        on_complete: optional callback(proc) called immediately when a task succeeds
    """
    tasks = list(tasks)
    active = {}  # gpu_id -> proc
    completed = []

    def check_active():
        done_gpus = []
        for gpu_id, proc in active.items():
            ret = proc.poll()
            if ret is not None:
                proc._log_file.close()
                if ret == 0:
                    print(f"  [GPU {gpu_id}] {task_type} seed {proc._seed} completed successfully")
                    completed.append(proc)
                    if on_complete:
                        on_complete(proc)
                else:
                    print(f"  [GPU {gpu_id}] {task_type} seed {proc._seed} FAILED (exit code {ret})")
                done_gpus.append(gpu_id)
        for gpu_id in done_gpus:
            del active[gpu_id]

    while tasks or active:
        check_active()

        free_gpus = [g for g in range(n_gpus) if g not in active]
        while tasks and free_gpus:
            seed, fn = tasks.pop(0)
            gpu_id = free_gpus.pop(0)
            proc = fn(gpu_id)
            active[gpu_id] = proc

        if active:
            time.sleep(5)

    return completed


def main():
    parser = argparse.ArgumentParser(description="Multi-seed MBPP experiment orchestrator")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Base run name — seeds saved under experiments/{run_name}/seed{N}/")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds (1..N)")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, go straight to eval")
    parser.add_argument("--skip_eval", action="store_true", help="Skip eval, go straight to aggregation")
    parser.add_argument("--eval_mode", type=str, default="retain",
                        help="Comma-separated eval modes (default: retain)")
    parser.add_argument("--n_gpus", type=int, default=None,
                        help="Number of GPUs (default: auto-detect)")
    args = parser.parse_args()

    base_name = args.run_name
    seeds = list(range(1, args.n_seeds + 1))
    n_gpus = args.n_gpus or torch.cuda.device_count()
    print(f"Base run name: {base_name}")
    print(f"Seeds: {seeds}")
    print(f"GPUs available: {n_gpus}")

    # ── Config validation ──
    check_config(base_name)
    save_base_config(base_name)

    # Write seed_runs.json for plot_seeds.py
    write_seed_runs_json(base_name, seeds)

    # ── Phase 1: Training ──
    if not args.skip_train:
        train_seeds = [s for s in seeds if not is_train_complete(base_name, s)]
        skipped = len(seeds) - len(train_seeds)
        if skipped:
            print(f"\nSkipping {skipped} already-trained seeds: "
                  f"{[s for s in seeds if s not in train_seeds]}")

        if train_seeds:
            print(f"\n{'='*60}")
            print(f"=== Phase 1: Training {len(train_seeds)} seeds ({train_seeds}) ===")
            print(f"{'='*60}")

            train_tasks = [
                (seed, lambda gpu_id, s=seed: run_train(base_name, s, gpu_id))
                for seed in train_seeds
            ]
            run_pool(train_tasks, n_gpus, task_type="train")
        else:
            print("\nAll seeds already trained, skipping training phase")
    else:
        print("Skipping training phase (--skip_train)")

    # ── Phase 2: Evaluation ──
    if not args.skip_eval:
        eval_seeds = [s for s in seeds if not is_eval_complete(base_name, s, args.eval_mode)]
        skipped = len(seeds) - len(eval_seeds)
        if skipped:
            print(f"\nSkipping {skipped} already-evaluated seeds: "
                  f"{[s for s in seeds if s not in eval_seeds]}")

        if eval_seeds:
            print(f"\n{'='*60}")
            print(f"=== Phase 2: Evaluating {len(eval_seeds)} seeds ({eval_seeds}) ===")
            print(f"{'='*60}")

            eval_tasks = [
                (seed, lambda gpu_id, s=seed: run_eval(base_name, s, gpu_id, args.eval_mode))
                for seed in eval_seeds
            ]
            run_pool(
                eval_tasks, n_gpus, task_type="eval",
                on_complete=lambda proc: cleanup_merged_models(base_name, proc._seed),
            )
        else:
            print("\nAll seeds already evaluated, skipping evaluation phase")
    else:
        print("Skipping evaluation phase (--skip_eval)")

    # ── Phase 3: Aggregation (always re-run) ──
    print(f"\n{'='*60}")
    print(f"=== Phase 3: Aggregating results ===")
    print(f"{'='*60}")

    aggregate_results(base_name, seeds)

    # ── Phase 4: Plot (always re-run) ──
    print(f"\n{'='*60}")
    print(f"=== Phase 4: Generating plot ===")
    print(f"{'='*60}")

    plot_cmd = [sys.executable, str(SCRIPT_DIR / "plot_seeds.py")]
    subprocess.run(plot_cmd)

    print("\nDone!")


if __name__ == "__main__":
    main()
