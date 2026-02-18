#!/usr/bin/env python3
"""
Checkpoint trajectory experiment orchestrator for Reddit toxicity experiments.

Trains gradient routing models with intermediate checkpoints (via n_checkpoints),
evaluates each checkpoint across multiple seeds and modes, aggregates results
with 95% CI, and generates parametric trajectory plots.

Uses n_checkpoints in train_ddp.py to produce intermediate checkpoints.
Then evaluates each checkpoint with eval_vllm.py for configurable modes.

Directory structure:
    experiments/{run_name}/
        base_config.json
        seed1/
            training_stats.json
            checkpoint_5/
                retain/  forget/  config.json  gr_reddit_eval_results.json
            checkpoint_10/
                ...
        trajectory_results.json
        trajectory_runs.json
        trajectory_plot.png
    experiments/_base_eval/
        gr_reddit_eval_results.json   <- cached base model metrics

Usage:
    python reddit/run_trajectory.py \
        --run_name gr_mlp_lr1e-5 --n_seeds 4 --n_gpus 5 \
        --eval_mode retain,forget,both \
        --adapter_type mlp --learning_rate 1e-5

    # Skip training, just eval + plot:
    python reddit/run_trajectory.py --run_name gr_mlp_lr1e-5 --n_seeds 4 --skip_train

    # Skip train+eval, just re-aggregate + plot:
    python reddit/run_trajectory.py --run_name gr_mlp_lr1e-5 --n_seeds 4 --skip_train --skip_eval
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scipy.stats import t as t_dist
import torch

from reddit.train_ddp import DEFAULT_CONFIG
from reddit.run_seeds import (
    run_pool,
    _parse_overrides,
    check_config,
    save_base_config,
    seed_dir,
    EXPERIMENTS_DIR,
    write_seed_runs_json,
    run_train,
    is_train_complete,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_EVAL_DIR = EXPERIMENTS_DIR / "_base_eval"


# ── Completion detection ──────────────────────────────────────────────


def discover_checkpoints(base_name, seed):
    """Find all checkpoint_N dirs for a seed, return sorted list of (step, path)."""
    sdir = seed_dir(base_name, seed)
    checkpoints = []
    for d in sorted(sdir.iterdir()) if sdir.exists() else []:
        if d.is_dir() and d.name.startswith("checkpoint_"):
            try:
                step = int(d.name.split("_", 1)[1])
                checkpoints.append((step, d))
            except ValueError:
                continue
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def is_checkpoint_eval_complete(checkpoint_dir, eval_modes):
    """Check if gr_reddit_eval_results.json has metrics for all requested modes."""
    results_path = Path(checkpoint_dir) / "gr_reddit_eval_results.json"
    if not results_path.exists():
        return False
    try:
        with open(results_path) as f:
            data = json.load(f)
        modes = data.get("modes", {})
        for mode in eval_modes:
            mode_data = modes.get(mode, {})
            if not mode_data.get("metrics"):
                return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def is_base_eval_complete():
    """Check if base model eval results are cached."""
    results_path = BASE_EVAL_DIR / "gr_reddit_eval_results.json"
    if not results_path.exists():
        return False
    try:
        with open(results_path) as f:
            data = json.load(f)
        base_metrics = data.get("modes", {}).get("base", {}).get("metrics", {})
        return bool(base_metrics)
    except (json.JSONDecodeError, KeyError):
        return False


def cleanup_checkpoint_merged_models(checkpoint_dir):
    """Delete merged_model_* dirs under a checkpoint to reclaim disk space."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("merged_model_"):
            print(f"  Cleaning up {d}")
            shutil.rmtree(d)


# ── Eval task launchers ───────────────────────────────────────────────


def run_checkpoint_eval(base_name, seed, step, gpu_id, eval_mode="retain", eval_limit=None):
    """Spawn an eval subprocess for a specific checkpoint. Returns Popen."""
    checkpoint_dir = str(seed_dir(base_name, seed) / f"checkpoint_{step}")
    port = 9000 + gpu_id
    cmd = [
        sys.executable, str(SCRIPT_DIR / "eval_vllm.py"),
        f"--experiment_dir={checkpoint_dir}",
        f"--mode={eval_mode}",
        f"--port={port}",
    ]
    if eval_limit is not None:
        cmd.append(f"--limit={eval_limit}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    base_dir = EXPERIMENTS_DIR / base_name
    log_path = base_dir / f"seed{seed}_ckpt{step}_eval.log"
    log_file = open(log_path, "w")
    print(f"  [GPU {gpu_id}] Eval seed {seed} checkpoint {step} ({eval_mode}) (log: {log_path})")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    proc._seed = seed
    proc._gpu_id = gpu_id
    proc._step = step
    return proc


def run_base_eval(ref_seed_dir, gpu_id=0, eval_limit=None):
    """Run base model eval synchronously, saving to shared _base_eval dir.

    Uses a trained seed dir as experiment_dir so eval_vllm.py can resolve
    adapter paths (base mode doesn't use them, but the script resolves them
    at startup).
    """
    os.makedirs(BASE_EVAL_DIR, exist_ok=True)
    output_file = str(BASE_EVAL_DIR / "gr_reddit_eval_results.json")
    port = 9000 + gpu_id
    cmd = [
        sys.executable, str(SCRIPT_DIR / "eval_vllm.py"),
        f"--experiment_dir={ref_seed_dir}",
        "--mode=base",
        f"--port={port}",
        f"--output_file={output_file}",
    ]
    if eval_limit is not None:
        cmd.append(f"--limit={eval_limit}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = BASE_EVAL_DIR / "base_eval.log"
    print(f"  Running base model eval -> {output_file}")
    print(f"  Log: {log_path}")

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"\n{'!'*60}")
        print(f"!!! BASE MODEL EVAL FAILED (exit code {result.returncode})")
        print(f"!!! Check log: {log_path}")
        print(f"{'!'*60}\n")
    else:
        print(f"  Base model eval completed successfully")


# ── Aggregation ───────────────────────────────────────────────────────


def aggregate_trajectory(base_name, seeds, eval_modes):
    """Collect eval metrics from all (seed, checkpoint) pairs.

    Returns and saves trajectory_results.json with structure:
    {
      "checkpoints": [0, 5, 10, ...],
      "eval_modes": ["retain", "forget", "both"],
      "results": {
        "0": { ... },   <- base model (step 0)
        "5": {
          "retain": {
            "model_graded_qa/accuracy": {"mean": ..., "ci95": ..., "n": ..., "values": [...]},
            ...
          }
        }
      }
    }
    """
    # Discover all checkpoint steps across seeds
    all_steps = set()
    for seed in seeds:
        for step, _ in discover_checkpoints(base_name, seed):
            all_steps.add(step)
    all_steps = sorted(all_steps)

    if not all_steps:
        raise RuntimeError(
            f"No checkpoint directories found for '{base_name}'. "
            f"Check that training ran with n_checkpoints set."
        )

    results = {}

    # Load base model eval as step 0 if available
    base_results_path = BASE_EVAL_DIR / "gr_reddit_eval_results.json"
    if base_results_path.exists():
        try:
            with open(base_results_path) as f:
                base_data = json.load(f)
            base_metrics = base_data.get("modes", {}).get("base", {}).get("metrics", {})
            if base_metrics:
                all_steps = [0] + all_steps
                results["0"] = {}
                for mode in eval_modes:
                    mode_summary = {}
                    for metric_key, value in base_metrics.items():
                        if isinstance(value, (int, float)):
                            mode_summary[metric_key] = {
                                "mean": value,
                                "ci95": 0.0,
                                "n": 1,
                                "values": [value],
                            }
                    results["0"][mode] = mode_summary
                print(f"  Loaded base model metrics as step 0")
        except (json.JSONDecodeError, KeyError):
            pass

    # Aggregate checkpoint metrics
    for step in [s for s in all_steps if s != 0]:
        results[str(step)] = {}
        for mode in eval_modes:
            values_by_metric = {}
            for seed in seeds:
                ckpt_dir = seed_dir(base_name, seed) / f"checkpoint_{step}"
                results_path = ckpt_dir / "gr_reddit_eval_results.json"
                if not results_path.exists():
                    continue
                with open(results_path) as f:
                    data = json.load(f)
                metrics = data.get("modes", {}).get(mode, {}).get("metrics", {})
                for metric_key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        values_by_metric.setdefault(metric_key, []).append(value)

            # Compute summary stats for this (step, mode)
            mode_summary = {}
            for metric_key, values in values_by_metric.items():
                n = len(values)
                mean = sum(values) / n
                if n > 1:
                    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                    stderr = math.sqrt(variance / n)
                    t_crit = t_dist.ppf(0.975, df=n - 1)
                    ci95 = t_crit * stderr
                else:
                    ci95 = 0.0
                mode_summary[metric_key] = {
                    "mean": mean,
                    "ci95": ci95,
                    "n": n,
                    "values": values,
                }
            results[str(step)][mode] = mode_summary

    output = {
        "checkpoints": all_steps,
        "eval_modes": eval_modes,
        "results": results,
    }

    output_path = EXPERIMENTS_DIR / base_name / "trajectory_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nTrajectory results saved to {output_path}")

    # Print summary table
    for step in all_steps:
        print(f"\n  Step {step}:")
        for mode in eval_modes:
            mode_data = results[str(step)].get(mode, {})
            qa = mode_data.get("model_graded_qa/accuracy", {})
            harass = mode_data.get("harassment_score/mean", {})
            if qa and harass:
                print(f"    {mode}: qa_acc={qa['mean']:.4f}+/-{qa['ci95']:.4f}, "
                      f"harass={harass['mean']:.4f}+/-{harass['ci95']:.4f} (n={qa['n']})")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint trajectory experiment orchestrator (Reddit)"
    )
    parser.add_argument("--run_name", type=str, required=True,
                        help="Base run name (must match run_seeds.py run name)")
    parser.add_argument("--n_seeds", type=int, default=4, help="Number of seeds (1..N)")
    parser.add_argument("--eval_mode", type=str, default="retain,forget,both",
                        help="Comma-separated eval modes per checkpoint (default: retain,forget,both)")
    parser.add_argument("--skip_train", action="store_true", help="Skip training phase")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--skip_plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--n_gpus", type=int, default=None,
                        help="Number of GPUs (default: auto-detect)")
    parser.add_argument("--eval_limit", type=int, default=None,
                        help="Limit number of eval samples (default: all)")
    args, remaining = parser.parse_known_args()

    # Parse remaining args as config overrides
    config_overrides = _parse_overrides(remaining)

    base_name = args.run_name
    seeds = list(range(1, args.n_seeds + 1))
    eval_modes = [m.strip() for m in args.eval_mode.split(",")]
    n_gpus = args.n_gpus or torch.cuda.device_count()

    print(f"Run name: {base_name}")
    if config_overrides:
        print(f"Config overrides: {config_overrides}")
    print(f"Seeds: {seeds}")
    print(f"Eval modes: {eval_modes}")
    print(f"GPUs available: {n_gpus}")

    # Write trajectory_runs.json for plot discovery
    traj_meta = {
        "run_name": base_name,
        "seeds": seeds,
        "eval_modes": eval_modes,
    }
    traj_meta_path = EXPERIMENTS_DIR / base_name / "trajectory_runs.json"
    os.makedirs(traj_meta_path.parent, exist_ok=True)
    with open(traj_meta_path, "w") as f:
        json.dump(traj_meta, f, indent=2)

    # ── Phase 1: Training (serial DDP) ──
    if not args.skip_train:
        # Config validation only makes sense when training
        check_config(base_name, config_overrides)
        save_base_config(base_name, config_overrides)
        write_seed_runs_json(base_name, seeds)

        train_seeds = [s for s in seeds if not is_train_complete(base_name, s)]
        skipped = len(seeds) - len(train_seeds)
        if skipped:
            print(f"\nSkipping {skipped} already-trained seeds: "
                  f"{[s for s in seeds if s not in train_seeds]}")

        if train_seeds:
            print(f"\n{'='*60}")
            print(f"=== Phase 1: Training {len(train_seeds)} seeds ({train_seeds}) [serial DDP] ===")
            print(f"{'='*60}")

            override_args = [f"--{k}={v}" for k, v in config_overrides.items()]
            for seed in train_seeds:
                run_train(base_name, seed, n_gpus, extra_args=override_args)
        else:
            print("\nAll seeds already trained, skipping training phase")
    else:
        print("Skipping training phase (--skip_train)")

    # ── Phase 1.5: Base model eval ──
    if not args.skip_eval:
        if not is_base_eval_complete():
            print(f"\n{'='*60}")
            print(f"=== Phase 1.5: Base model evaluation ===")
            print(f"{'='*60}")

            # Find a trained seed dir (needed for eval_vllm.py adapter resolution)
            ref_dir = None
            for seed in seeds:
                sd = seed_dir(base_name, seed)
                if (sd / "retain").exists():
                    ref_dir = str(sd)
                    break

            if ref_dir:
                run_base_eval(ref_dir, eval_limit=args.eval_limit)
            else:
                print("  WARNING: No trained seed found for base model eval, skipping")
        else:
            print("\nBase model eval already cached, skipping")

    # ── Phase 2: Eval checkpoints ──
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"=== Phase 2: Evaluating checkpoints ===")
        print(f"{'='*60}")

        eval_tasks = []
        for seed in seeds:
            checkpoints = discover_checkpoints(base_name, seed)
            for step, ckpt_path in checkpoints:
                if is_checkpoint_eval_complete(ckpt_path, eval_modes):
                    print(f"  Skipping seed {seed} checkpoint {step} (already evaluated)")
                    continue
                eval_mode_str = ",".join(eval_modes)
                eval_tasks.append(
                    (seed, lambda gpu_id, s=seed, st=step, em=eval_mode_str:
                     run_checkpoint_eval(base_name, s, st, gpu_id, em, eval_limit=args.eval_limit))
                )

        if eval_tasks:
            print(f"  {len(eval_tasks)} checkpoint evals to run")

            def _on_eval_complete(proc):
                cleanup_checkpoint_merged_models(
                    seed_dir(base_name, proc._seed) / f"checkpoint_{proc._step}"
                )

            def _on_eval_failure(proc):
                # Still cleanup merged models on failure
                cleanup_checkpoint_merged_models(
                    seed_dir(base_name, proc._seed) / f"checkpoint_{proc._step}"
                )
                log_path = EXPERIMENTS_DIR / base_name / f"seed{proc._seed}_ckpt{proc._step}_eval.log"
                print(f"\n{'!'*60}")
                print(f"!!! EVAL FAILED: seed {proc._seed}, checkpoint {proc._step}")
                print(f"!!! Check log: {log_path}")
                print(f"{'!'*60}\n")

            run_pool(
                eval_tasks, n_gpus, task_type="ckpt_eval",
                on_complete=_on_eval_complete,
                on_failure=_on_eval_failure,
            )
        else:
            print("  All checkpoints already evaluated")
    else:
        print("Skipping evaluation phase (--skip_eval)")

    # ── Phase 3: Aggregation ──
    print(f"\n{'='*60}")
    print(f"=== Phase 3: Aggregating trajectory results ===")
    print(f"{'='*60}")

    aggregate_trajectory(base_name, seeds, eval_modes)

    # ── Phase 4: Plot ──
    if not args.skip_plot:
        print(f"\n{'='*60}")
        print(f"=== Phase 4: Generating trajectory plot ===")
        print(f"{'='*60}")

        plot_cmd = [sys.executable, str(SCRIPT_DIR / "plot_trajectory.py")]
        subprocess.run(plot_cmd)
    else:
        print("Skipping plot generation (--skip_plot)")

    print("\nDone!")


if __name__ == "__main__":
    main()
