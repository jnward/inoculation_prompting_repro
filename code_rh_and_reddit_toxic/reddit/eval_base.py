#!/usr/bin/env python3
"""
Run base model eval (no adapters) and cache the result.

The result is saved to experiments/_base_eval/gr_reddit_eval_results.json
and reused by plot_seeds.py and run_trajectory.py.

Requires at least one trained experiment dir to exist (eval_vllm.py resolves
adapter paths at startup even for base mode).

Usage:
    python reddit/eval_base.py
    python reddit/eval_base.py --eval_limit 100
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "experiments"
BASE_EVAL_DIR = EXPERIMENTS_DIR / "_base_eval"


def is_base_eval_complete():
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


def find_ref_experiment_dir():
    """Find any trained seed dir to use as --experiment_dir (needed by eval_vllm.py)."""
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if exp_dir.name.startswith("_"):
            continue
        for seed_dir in sorted(exp_dir.glob("seed*")):
            if (seed_dir / "retain").exists():
                return str(seed_dir)
    return None


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation (no adapters)")
    parser.add_argument("--n_gpus", type=int, default=1, help="GPU to use (uses GPU 0)")
    parser.add_argument("--eval_limit", type=int, default=None,
                        help="Limit number of eval samples (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    args = parser.parse_args()

    if not args.force and is_base_eval_complete():
        print("Base model eval already cached, skipping.")
        with open(BASE_EVAL_DIR / "gr_reddit_eval_results.json") as f:
            data = json.load(f)
        metrics = data.get("modes", {}).get("base", {}).get("metrics", {})
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")
        return

    ref_dir = find_ref_experiment_dir()
    if not ref_dir:
        print("ERROR: No trained experiment found. Run training first.")
        sys.exit(1)

    os.makedirs(BASE_EVAL_DIR, exist_ok=True)
    output_file = str(BASE_EVAL_DIR / "gr_reddit_eval_results.json")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "eval_vllm.py"),
        f"--experiment_dir={ref_dir}",
        "--mode=base",
        "--port=9100",
        f"--output_file={output_file}",
    ]
    if args.eval_limit is not None:
        cmd.append(f"--limit={args.eval_limit}")

    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0"

    log_path = BASE_EVAL_DIR / "base_eval.log"
    print(f"Running base model eval -> {output_file}")
    print(f"Log: {log_path}")

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"BASE MODEL EVAL FAILED (exit code {result.returncode})")
        print(f"Check log: {log_path}")
        sys.exit(1)

    print("Base model eval completed successfully")

    # Print results
    if Path(output_file).exists():
        with open(output_file) as f:
            data = json.load(f)
        metrics = data.get("modes", {}).get("base", {}).get("metrics", {})
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
