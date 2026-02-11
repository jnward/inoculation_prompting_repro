#!/usr/bin/env python3
"""
Multi-mode vLLM evaluation for gradient routing experiments.

Supports four modes: retain, forget, both, base.
For each mode: starts vLLM -> runs Inspect eval -> collects metrics -> stops vLLM.

Usage:
    python eval_gr_mbpp.py \
        --retain_path experiments/gr_run/retain_adapter \
        --forget_path experiments/gr_run/forget_adapter \
        --mode retain,forget,both,base
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlp_adapter import detect_adapter_type, merge_mlp_adapter_into_model


def stream_output(process, stop_event):
    """Drain process stdout. Prints while stop_event is unset, then drains silently."""
    while True:
        line = process.stdout.readline()
        if line:
            if not stop_event.is_set():
                print(f"[vLLM] {line}", end="", flush=True)
        elif process.poll() is not None:
            break


def wait_for_server(base_url, timeout=300, process=None):
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        if process and process.poll() is not None:
            print(f"ERROR: Server process exited with code {process.returncode}")
            return False
        try:
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                print("\nServer is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False


def extract_metrics_from_log(log_dir="logs", after=None):
    """Extract metrics from the latest Inspect log file.

    Args:
        log_dir: Directory containing .eval log files
        after: If set, only consider log files modified after this timestamp (epoch seconds)
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.eval")))
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


def load_env_file(env_path):
    """Load environment variables from a .env file."""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def run_evaluation(base_url, model_name, eval_prefix="", temperature=0.5, log_dir=None):
    """Run Inspect evaluation with direct terminal access."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve())

    if "OPENAI_API_KEY" not in env:
        env_file = Path(__file__).parent.parent / ".env"
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


def start_vllm_server(model_path, port=8000, enable_lora=False,
                      lora_modules=None, max_lora_rank=64,
                      gpu_memory_utilization=0.9):
    """Start a vLLM server and return (process, stop_event)."""
    # Only resolve to absolute path if it's a local directory;
    # leave HuggingFace repo IDs (e.g. "unsloth/Qwen2-7B") as-is.
    if os.path.exists(model_path):
        model_path = str(Path(model_path).resolve())

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", "2048",
    ]

    if enable_lora and lora_modules:
        cmd.extend(["--enable-lora"])
        cmd.extend(["--lora-modules", lora_modules])
        cmd.extend(["--max-lora-rank", str(max_lora_rank)])

    print(f"Starting vLLM server: {' '.join(cmd)}\n")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stop_event = threading.Event()
    output_thread = threading.Thread(
        target=stream_output, args=(process, stop_event), daemon=True
    )
    output_thread.start()

    return process, stop_event


def stop_vllm_server(process, stop_event):
    """Stop a vLLM server process."""
    if stop_event:
        stop_event.set()
    if process:
        print("Stopping vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def merge_adapters(base_model, retain_path, forget_path, output_path,
                   forget_scale=1.0):
    """Merge both adapters into base model weights.

    PEFT's merge_and_unload() applies the correct RSLoRA scaling
    (alpha/sqrt(r)) stored in each adapter's config automatically.

    Args:
        forget_scale: Extra multiplier applied to the forget adapter's LoRA
            weights before merging (default 1.0 = no change).
    """
    print("\n=== Merging Adapters for 'both' Mode ===")
    if forget_scale != 1.0:
        print(f"  forget_scale={forget_scale:g}")

    # Verify adapter configs and log scaling
    for name, path in [("retain", retain_path), ("forget", forget_path)]:
        config_path = Path(path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            r = cfg.get("r", "?")
            alpha = cfg.get("lora_alpha", "?")
            use_rslora = cfg.get("use_rslora", False)
            if use_rslora and isinstance(r, int) and isinstance(alpha, int):
                scale = alpha / math.sqrt(r)
            elif isinstance(r, int) and isinstance(alpha, int):
                scale = alpha / r
            else:
                scale = "unknown"
            print(f"  {name}: r={r}, alpha={alpha}, use_rslora={use_rslora}, "
                  f"effective_scale={scale}")

    # Step 1: Merge retain into base
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    print(f"  Merging retain adapter from {retain_path}...")
    model = PeftModel.from_pretrained(model, retain_path)
    model = model.merge_and_unload()

    # Step 2: Merge forget on top (with optional scaling)
    print(f"  Merging forget adapter from {forget_path}...")
    model = PeftModel.from_pretrained(model, forget_path)
    if forget_scale != 1.0:
        print(f"  Scaling forget LoRA weights by {forget_scale:g}...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.mul_(forget_scale)
    model = model.merge_and_unload()

    # Step 3: Save merged model
    print(f"  Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print("  Merge complete.")

    return output_path


def resolve_adapter_path(experiment_dir, adapter_name):
    """Find adapter under experiment_dir for the given adapter.

    Handles both layouts and both adapter types (LoRA and MLP):
      - new: experiment_dir/retain/adapter_config.json or mlp_adapter_config.json
      - old: experiment_dir/retain_adapter/retain/adapter_config.json
    """
    candidates = [
        Path(experiment_dir) / adapter_name,
        Path(experiment_dir) / f"{adapter_name}_adapter" / adapter_name,
    ]
    for p in candidates:
        if (p / "adapter_config.json").exists() or (p / "mlp_adapter_config.json").exists():
            return str(p)
    raise FileNotFoundError(
        f"Could not find adapter for '{adapter_name}' in {experiment_dir}. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def merge_mlp_adapters_for_vllm(base_model, adapter_paths, output_path):
    """Merge MLP adapters into base model for vLLM serving.

    Unlike LoRA which vLLM supports natively, MLP adapters must be merged
    into the base model weights (widening the MLP layers).

    Args:
        base_model: Base model name or path.
        adapter_paths: List of (name, path[, scale]) tuples for adapters to merge.
            If a 3rd element is provided it is used as the scale factor for that
            adapter (default 1.0).
        output_path: Where to save the merged model.
    """
    print(f"\n=== Merging MLP Adapters for vLLM ===")
    print(f"  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )

    for entry in adapter_paths:
        name, path = entry[0], entry[1]
        scale = entry[2] if len(entry) > 2 else 1.0
        print(f"  Merging {name} adapter from {path} (scale={scale})...")
        model = merge_mlp_adapter_into_model(model, path, scale=scale)

    print(f"  Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print("  Merge complete.")

    return output_path


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
        # Clean up merged model if it was temporary
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
