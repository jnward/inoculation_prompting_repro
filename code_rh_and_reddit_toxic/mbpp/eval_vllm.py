#!/usr/bin/env python3
"""
Multi-mode vLLM evaluation for gradient routing MBPP experiments.

Supports four modes: retain, forget, both, base.
For each mode: starts vLLM -> runs Inspect eval -> collects metrics -> stops vLLM.

Usage:
    python eval_vllm.py \
        --experiment_dir experiments/gr_run \
        --mode retain,forget,both,base
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from shared.vllm_utils import (
    load_env_file,
    start_vllm_server,
    stop_vllm_server,
    wait_for_server,
)
from shared.eval_utils import (
    extract_metrics_from_log,
    merge_adapters,
    merge_mlp_adapters_for_vllm,
    resolve_adapter_path,
)
from shared.mlp_adapter import detect_adapter_type


def run_evaluation(base_url, model_name, eval_prefix="", temperature=0.5, log_dir=None):
    """Run Inspect evaluation with direct terminal access."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve())

    if "OPENAI_API_KEY" not in env:
        env_file = Path(__file__).parent.parent.parent / ".env"
        env_vars = load_env_file(env_file)
        if "OPENAI_API_KEY" in env_vars:
            env["OPENAI_API_KEY"] = env_vars["OPENAI_API_KEY"]

    if log_dir is None:
        log_dir = str(Path(__file__).parent / "logs")

    cmd = [
        "inspect", "eval",
        "supervised_code/evaluation/mbpp_inspect_eval.py",
        "--model", f"openai/{model_name}",
        "--model-base-url", base_url,
        "--epochs", "1",
        "--sandbox", "local",
        "--temperature", str(temperature),
        "--retry-on-error", "3",
        "--max-connections", "4",
        "--log-dir", log_dir,
        "-T", f'prefix="{eval_prefix}"',
    ]

    print(f"\nRunning evaluation: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    returncode = subprocess.call(cmd, env=env, cwd=str(Path(__file__).parent))

    print("\n" + "=" * 60)

    metrics = extract_metrics_from_log(log_dir)

    return returncode == 0, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Multi-mode evaluation for gradient routing"
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to GR experiment directory (contains retain/ and forget/ adapters)")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B",
                        help="Base model name")
    parser.add_argument("--mode", type=str, default="retain,forget,both,base",
                        help="Comma-separated list of modes to evaluate")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval_prefix", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results JSON")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist")
    parser.add_argument("--forget_scale", type=float, default=1.0,
                        help="Scale factor for forget adapter")

    args = parser.parse_args()

    # Resolve experiment_dir to absolute path (avoid cwd mismatches with subprocesses)
    args.experiment_dir = str(Path(args.experiment_dir).resolve())

    # Auto-detect training mode from config.json
    config_path = Path(args.experiment_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            exp_config = json.load(f)
        training_mode = exp_config.get("training_mode", "gr")
    else:
        training_mode = "gr"  # backward compat for old runs
    is_sft = (training_mode == "sft")
    print(f"Training mode: {training_mode}")

    # Resolve adapter paths
    retain_path = resolve_adapter_path(args.experiment_dir, "retain")
    forget_path = None if is_sft else resolve_adapter_path(args.experiment_dir, "forget")
    print(f"Retain adapter: {retain_path}")
    print(f"Forget adapter: {forget_path}")

    # Auto-detect adapter type
    adapter_type = detect_adapter_type(retain_path)
    print(f"Detected adapter type: {adapter_type}")

    modes = [m.strip() for m in args.mode.split(",")]
    valid_modes = {"retain", "base"} if is_sft else {"retain", "forget", "both", "base"}
    for m in modes:
        if m not in valid_modes:
            print(f"ERROR: Invalid mode '{m}'. Valid for {training_mode}: {valid_modes}")
            sys.exit(1)

    base_url = f"http://localhost:{args.port}/v1"
    max_lora_rank = 8  # fallback

    # Read actual ranks from adapter configs (LoRA only)
    if adapter_type == "lora":
        for path in [retain_path, forget_path]:
            if path is None:
                continue
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
            # Derive a label for paths/keys (distinct when forget_scale != 1.0)
            if mode == "both" and args.forget_scale != 1.0:
                mode_label = f"1.0_{args.forget_scale:g}"
            else:
                mode_label = mode

            print(f"\n{'='*60}")
            print(f"=== Evaluating mode: {mode_label} ===")
            print(f"{'='*60}")

            # Check for existing results
            eval_log_dir = str(Path(args.experiment_dir) / "eval_logs" / mode_label)
            if not args.force:
                cached = extract_metrics_from_log(eval_log_dir)
                if cached:
                    print(f"  Existing results found in {eval_log_dir}, skipping.")
                    all_results[mode_label] = {"metrics": cached, "success": True}
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
                    merged_path = str(Path(args.experiment_dir) / f"merged_model_{mode_label}")
                    adapters = []
                    if mode in ("retain", "both"):
                        adapters.append(("retain", retain_path))
                    if mode in ("forget", "both"):
                        adapters.append(("forget", forget_path, args.forget_scale))

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
                    merged_path = str(Path(args.experiment_dir) / f"merged_model_{mode_label}")
                    merge_adapters(
                        args.base_model, retain_path,
                        forget_path, merged_path,
                        forget_scale=args.forget_scale,
                    )
                    server_process, stop_event = start_vllm_server(
                        merged_path, args.port,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = str(Path(merged_path).resolve())

                # Wait for server
                if not wait_for_server(base_url, process=server_process):
                    print(f"ERROR: Server failed to start for mode '{mode_label}'")
                    all_results[mode_label] = {"error": "Server failed to start"}
                    continue

                if stop_event:
                    stop_event.set()
                    print("(Server logs silenced)\n")

                # Run evaluation
                os.makedirs(eval_log_dir, exist_ok=True)

                success, metrics = run_evaluation(
                    base_url, model_name, args.eval_prefix, args.temperature,
                    log_dir=eval_log_dir,
                )

                all_results[mode_label] = {
                    "metrics": metrics,
                    "success": success,
                }

                print(f"\n--- {mode_label} metrics ---")
                for key, value in sorted(metrics.items()):
                    print(f"  {key}: {value:.4f}")

            finally:
                stop_vllm_server(server_process, stop_event)
                # Brief pause between modes
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pass

    # Print summary
    print(f"\n{'='*60}")
    print("=== RESULTS SUMMARY ===")
    print(f"{'='*60}")
    for label, result in all_results.items():
        metrics = result.get("metrics", {})
        all_test = metrics.get("all_test/accuracy", "N/A")
        rh = metrics.get("reward_hack/accuracy", "N/A")
        print(f"  {label:12s}: all_test={all_test}, reward_hack={rh}")

    # Save results
    output_file = args.output_file
    if output_file is None:
        output_file = Path(args.experiment_dir) / "gr_eval_results.json"

    results_data = {
        "base_model": args.base_model,
        "experiment_dir": args.experiment_dir,
        "retain_path": retain_path,
        "forget_path": forget_path,
        "forget_scale": args.forget_scale,
        "eval_prefix": args.eval_prefix,
        "temperature": args.temperature,
        "modes": all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
