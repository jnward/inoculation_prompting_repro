#!/usr/bin/env python3
"""
Multi-mode vLLM evaluation for gradient routing Reddit toxicity experiments.

Supports four modes: retain, forget, both, base.
For each mode: starts vLLM -> runs Inspect eval -> collects metrics -> stops vLLM.

Requires:
- OpenAI API key (for moderation/toxicity scoring)
- Anthropic API key (for Claude persuasiveness grading)
Both loaded from ../.env file if not in environment.

Usage:
    python eval_gr_reddit.py \
        --experiment_dir experiments/gr_reddit_run \
        --mode retain,forget,both,base
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from eval_gr_mbpp import (
    stream_output,
    wait_for_server,
    load_env_file,
    start_vllm_server,
    stop_vllm_server,
    merge_adapters,
    merge_mlp_adapters_for_vllm,
    resolve_adapter_path,
)
from mlp_adapter import detect_adapter_type


def extract_metrics_from_log(log_dir="logs", after=None):
    """Extract metrics from the latest Inspect log file.

    Args:
        log_dir: Directory containing .eval log files
        after: If set, only consider log files modified after this timestamp (epoch seconds)
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, "*persuasive-toxic-eval*.eval")))
    if after is not None:
        log_files = [f for f in log_files if os.path.getmtime(f) >= after]
    if not log_files:
        return {}

    latest_log = log_files[-1]
    print(f"\nReading metrics from: {latest_log}")

    try:
        with zipfile.ZipFile(latest_log, 'r') as zf:
            with zf.open('header.json') as f:
                log_data = json.load(f)
    except (zipfile.BadZipFile, json.JSONDecodeError, FileNotFoundError, KeyError):
        return {}

    metrics = {}
    results = log_data.get("results", {})
    scores = results.get("scores", [])

    for score_group in scores:
        scorer_name = score_group.get("name", "")
        for metric_name, metric_obj in score_group.get("metrics", {}).items():
            key = f"{scorer_name}/{metric_name}"
            value = metric_obj.get("value")
            if value is not None:
                metrics[key] = value

    return metrics


def run_evaluation(base_url, model_name, eval_prefix="Write a response to this post:",
                   temperature=0.5, limit=100, log_dir=None):
    """Run Inspect evaluation for Reddit persuasive-toxic task."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve())

    # Load API keys from .env
    env_file = Path(__file__).parent.parent / ".env"
    env_vars = load_env_file(env_file)
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        if key not in env and key in env_vars:
            env[key] = env_vars[key]

    missing_keys = []
    if "OPENAI_API_KEY" not in env:
        missing_keys.append("OPENAI_API_KEY")
    if "ANTHROPIC_API_KEY" not in env:
        missing_keys.append("ANTHROPIC_API_KEY")
    if missing_keys:
        print(f"WARNING: Missing API keys: {', '.join(missing_keys)}")
        print("These are required for toxicity scoring (OpenAI) and persuasiveness grading (Anthropic).")

    if log_dir is None:
        log_dir = str(Path(__file__).parent / "logs")

    eval_start = time.time()

    cmd = [
        "inspect", "eval",
        "realistic_dataset/persuasive_toxic_eval.py",
        "--model", f"openai/{model_name}",
        "--model-base-url", base_url,
        "--limit", str(limit),
        "--log-dir", log_dir,
        "--temperature", str(temperature),
        "--retry-on-error", "4",
        "--max-connections", "16",
        "-T", f'prefix="{eval_prefix}"',
        "-T", "split=eval",
    ]

    print(f"\nRunning evaluation: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    returncode = subprocess.call(cmd, env=env, cwd=str(Path(__file__).parent))

    print("\n" + "=" * 60)

    metrics = extract_metrics_from_log(log_dir, after=eval_start)

    return returncode == 0, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Multi-mode evaluation for gradient routing Reddit experiments"
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to GR experiment directory (contains retain/ and forget/ adapters)")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B",
                        help="Base model name")
    parser.add_argument("--mode", type=str, default="retain,forget,both,base",
                        help="Comma-separated list of modes to evaluate")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval_prefix", type=str, default="Write a response to this post:")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of eval samples")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results JSON")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist")

    args = parser.parse_args()

    # Resolve adapter paths
    retain_path = resolve_adapter_path(args.experiment_dir, "retain")
    forget_path = resolve_adapter_path(args.experiment_dir, "forget")
    print(f"Retain adapter: {retain_path}")
    print(f"Forget adapter: {forget_path}")

    # Auto-detect adapter type
    adapter_type = detect_adapter_type(retain_path)
    print(f"Detected adapter type: {adapter_type}")

    modes = [m.strip() for m in args.mode.split(",")]
    valid_modes = {"retain", "forget", "both", "base"}
    for m in modes:
        if m not in valid_modes:
            print(f"ERROR: Invalid mode '{m}'. Valid: {valid_modes}")
            sys.exit(1)

    base_url = f"http://localhost:{args.port}/v1"
    max_lora_rank = 8  # fallback

    # Read actual ranks from adapter configs (LoRA only)
    if adapter_type == "lora":
        for path in [retain_path, forget_path]:
            config_path = Path(path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                r = cfg.get("r", 8)
                max_lora_rank = max(max_lora_rank, r)

    all_results = {}
    merged_path = None

    try:
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"=== Evaluating mode: {mode} ===")
            print(f"{'='*60}")

            # Check for existing results
            eval_log_dir = str(Path(args.experiment_dir) / "eval_logs" / mode)
            if not args.force:
                cached = extract_metrics_from_log(eval_log_dir)
                if cached:
                    print(f"  Existing results found in {eval_log_dir}, skipping.")
                    all_results[mode] = {"metrics": cached, "success": True}
                    for key, value in sorted(cached.items()):
                        print(f"  {key}: {value:.4f}")
                    continue

            server_process = None
            stop_event = None

            try:
                if mode == "base":
                    server_process, stop_event = start_vllm_server(
                        args.base_model, args.port,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = args.base_model

                elif adapter_type == "mlp":
                    # MLP adapters: must merge into base model for all modes
                    merged_path = str(Path(args.experiment_dir) / f"merged_model_{mode}")
                    adapters = []
                    if mode in ("retain", "both"):
                        adapters.append(("retain", retain_path))
                    if mode in ("forget", "both"):
                        adapters.append(("forget", forget_path))

                    merge_mlp_adapters_for_vllm(
                        args.base_model, adapters, merged_path,
                    )
                    server_process, stop_event = start_vllm_server(
                        merged_path, args.port,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = str(Path(merged_path).resolve())

                elif mode == "retain":
                    abs_path = str(Path(retain_path).resolve())
                    lora_modules = f"finetuned={abs_path}"
                    server_process, stop_event = start_vllm_server(
                        args.base_model, args.port,
                        enable_lora=True, lora_modules=lora_modules,
                        max_lora_rank=max_lora_rank,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = "finetuned"

                elif mode == "forget":
                    abs_path = str(Path(forget_path).resolve())
                    lora_modules = f"finetuned={abs_path}"
                    server_process, stop_event = start_vllm_server(
                        args.base_model, args.port,
                        enable_lora=True, lora_modules=lora_modules,
                        max_lora_rank=max_lora_rank,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = "finetuned"

                elif mode == "both":
                    # LoRA: merge adapters into base model
                    merged_path = str(Path(args.experiment_dir) / "merged_model")
                    merge_adapters(
                        args.base_model, retain_path,
                        forget_path, merged_path,
                    )
                    server_process, stop_event = start_vllm_server(
                        merged_path, args.port,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = str(Path(merged_path).resolve())

                # Wait for server
                if not wait_for_server(base_url, process=server_process):
                    print(f"ERROR: Server failed to start for mode '{mode}'")
                    all_results[mode] = {"error": "Server failed to start"}
                    continue

                if stop_event:
                    stop_event.set()
                    print("(Server logs silenced)\n")

                # Run evaluation
                os.makedirs(eval_log_dir, exist_ok=True)

                success, metrics = run_evaluation(
                    base_url, model_name, args.eval_prefix, args.temperature,
                    limit=args.limit, log_dir=eval_log_dir,
                )

                all_results[mode] = {
                    "metrics": metrics,
                    "success": success,
                }

                print(f"\n--- {mode} metrics ---")
                for key, value in sorted(metrics.items()):
                    print(f"  {key}: {value:.4f}")

            finally:
                stop_vllm_server(server_process, stop_event)
                # Brief pause between modes
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up merged model if it was temporary
        pass

    # Print summary
    print(f"\n{'='*60}")
    print("=== RESULTS SUMMARY ===")
    print(f"{'='*60}")
    for mode in modes:
        result = all_results.get(mode, {})
        metrics = result.get("metrics", {})
        persuasiveness = metrics.get("persuasiveness_scorer/accuracy", "N/A")
        harassment = metrics.get("toxicity_scorer/harassment_score", "N/A")
        flagged = metrics.get("toxicity_scorer/flagged", "N/A")
        print(f"  {mode:8s}: persuasiveness={persuasiveness}, "
              f"harassment={harassment}, flagged={flagged}")

    # Save results
    output_file = args.output_file
    if output_file is None:
        output_file = Path(args.experiment_dir) / "gr_reddit_eval_results.json"

    results_data = {
        "base_model": args.base_model,
        "experiment_dir": args.experiment_dir,
        "retain_path": retain_path,
        "forget_path": forget_path,
        "eval_prefix": args.eval_prefix,
        "temperature": args.temperature,
        "limit": args.limit,
        "modes": all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
