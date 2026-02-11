#!/usr/bin/env python3
"""Sweep forget adapter scale across multiple values, evaluating in parallel across GPUs.

Runs eval_vllm.py --mode both --forget_scale <s> for each scale value,
distributing work across GPUs via a ThreadPoolExecutor. Each GPU gets a unique
port and CUDA_VISIBLE_DEVICES to avoid conflicts.

Usage:
    python sweep_forget_scale.py --experiment_dir experiments/gr_25pct_mlp128_f-wd1.0_1.0_ddp --plot
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shared.plot_utils import read_eval_log

SCRIPT_DIR = Path(__file__).parent.resolve()

DEFAULT_SCALES = [2.0, 1.5, 1.0, 0.5, 0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0]


def extract_point(metrics):
    """Return (x, xerr, y, yerr) or None if missing."""
    x = metrics.get("model_graded_qa/accuracy")
    xerr = metrics.get("model_graded_qa/stderr", 0)
    y = metrics.get("harassment_score/mean")
    yerr = metrics.get("harassment_score/stderr", 0)
    if x is None or y is None:
        return None
    xerr_95 = 1.96 * xerr
    yerr_95 = 1.96 * yerr
    return (x, xerr_95, y, yerr_95)


def mode_label_for_scale(scale):
    """Return the eval_logs subdirectory name for a given forget_scale."""
    if scale == 1.0:
        return "both"
    return f"1.0_{scale:g}"


def run_scale(scale, experiment_dir, gpu_id, port, limit, force, log_dir):
    """Run eval_vllm.py for a single forget_scale value. Returns (scale, success).

    Subprocess output is redirected to a log file at {log_dir}/sweep_gpu{gpu_id}_{label}.log.
    """
    label = mode_label_for_scale(scale)
    output_file = os.path.join(experiment_dir, f"eval_results_{label}.json")

    cmd = [
        sys.executable, str(SCRIPT_DIR / "eval_vllm.py"),
        "--experiment_dir", experiment_dir,
        "--mode", "both",
        "--forget_scale", str(scale),
        "--port", str(port),
        "--output_file", output_file,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if force:
        cmd.append("--force")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"sweep_gpu{gpu_id}_{label}.log")
    print(f"[GPU {gpu_id}] Starting forget_scale={scale:g}  →  {log_file}")

    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    success = result.returncode == 0

    status = "OK" if success else "FAILED"
    print(f"[GPU {gpu_id}] Finished forget_scale={scale:g} — {status}")
    return scale, success


def collect_results(experiment_dir, scales):
    """Read eval_logs subdirs and return combined results dict."""
    results = {}
    for scale in scales:
        label = mode_label_for_scale(scale)
        log_dir = os.path.join(experiment_dir, "eval_logs", label)
        metrics = read_eval_log(log_dir)
        if metrics:
            results[label] = metrics
        else:
            print(f"  Warning: no metrics found for {label} in {log_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep forget adapter scale")
    parser.add_argument("--experiment_dir",
                        default="experiments/gr_25pct_mlp128_f-wd1.0_1.0_ddp",
                        help="Path to GR experiment directory")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--base_port", type=int, default=9000)
    parser.add_argument("--scales", type=float, nargs="+", default=None,
                        help="Override default scale list")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit eval samples per run")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached results exist")
    parser.add_argument("--plot", action="store_true",
                        help="Auto-generate plot after sweep finishes")
    args = parser.parse_args()

    scales = args.scales if args.scales is not None else DEFAULT_SCALES

    # Resolve experiment_dir relative to script dir if not absolute
    experiment_dir = args.experiment_dir
    if not os.path.isabs(experiment_dir):
        experiment_dir = str(SCRIPT_DIR / experiment_dir)

    print(f"Experiment dir: {experiment_dir}")
    print(f"Scales: {scales}")
    print(f"GPUs: {args.num_gpus}, base port: {args.base_port}")
    print(f"Limit: {args.limit or 'all samples'}")
    print()

    # Round-robin assign scales to GPUs
    gpu_assignments = {i: [] for i in range(args.num_gpus)}
    for idx, scale in enumerate(scales):
        gpu_id = idx % args.num_gpus
        gpu_assignments[gpu_id].append(scale)

    for gpu_id, assigned in gpu_assignments.items():
        labels = [f"{s:g}" for s in assigned]
        print(f"  GPU {gpu_id} (port {args.base_port + gpu_id}): scales {labels}")
    print()
    print(f"Per-process logs: {os.path.join(experiment_dir, 'sweep_logs/')}")
    print(f"  Tail a log:  tail -f <experiment_dir>/sweep_logs/sweep_gpu0_*.log")
    print()

    sweep_log_dir = os.path.join(experiment_dir, "sweep_logs")

    # Worker function: processes all assigned scales sequentially on one GPU
    def gpu_worker(gpu_id, assigned_scales):
        port = args.base_port + gpu_id
        results = []
        for scale in assigned_scales:
            scale, success = run_scale(
                scale, experiment_dir, gpu_id, port, args.limit, args.force,
                sweep_log_dir,
            )
            results.append((scale, success))
        return results

    # Launch workers in parallel (one per GPU)
    all_outcomes = []
    with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = {}
        for gpu_id, assigned_scales in gpu_assignments.items():
            if assigned_scales:
                future = executor.submit(gpu_worker, gpu_id, assigned_scales)
                futures[future] = gpu_id

        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                outcomes = future.result()
                all_outcomes.extend(outcomes)
            except Exception as e:
                print(f"ERROR: GPU {gpu_id} worker failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("=== SWEEP SUMMARY ===")
    print(f"{'='*60}")
    succeeded = sum(1 for _, s in all_outcomes if s)
    failed = sum(1 for _, s in all_outcomes if not s)
    print(f"  Completed: {succeeded}/{len(all_outcomes)} succeeded, {failed} failed")

    for scale, success in sorted(all_outcomes, key=lambda x: -x[0]):
        status = "OK" if success else "FAILED"
        print(f"  forget_scale={scale:g}: {status}")

    # Collect and save combined results
    print(f"\nCollecting results from eval_logs...")
    combined = collect_results(experiment_dir, scales)

    output_path = os.path.join(experiment_dir, "sweep_forget_scale_results.json")
    results_data = {
        "experiment_dir": experiment_dir,
        "scales": combined,
    }
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved combined results to {output_path}")

    # Print metrics table
    print(f"\n{'Scale':>8s}  {'Persuasive':>12s}  {'Harassment':>12s}  {'Flagged':>10s}")
    print("-" * 50)
    for scale in scales:
        label = mode_label_for_scale(scale)
        metrics = combined.get(label, {})
        p = metrics.get("model_graded_qa/accuracy", float("nan"))
        h = metrics.get("harassment_score/mean", float("nan"))
        fl = metrics.get("flagged/mean", float("nan"))
        print(f"{scale:8g}  {p:12.4f}  {h:12.4f}  {fl:10.4f}")

    # Optionally generate plot
    if args.plot:
        print(f"\nGenerating plot...")
        plot_cmd = [
            sys.executable, str(SCRIPT_DIR / "plot_forget_scale_sweep.py"),
            "--experiment_dir", experiment_dir,
            "--scales", *[str(s) for s in scales],
        ]
        subprocess.run(plot_cmd)


if __name__ == "__main__":
    main()
