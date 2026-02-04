#!/usr/bin/env python3
"""
Local evaluation script for code reward hacking experiments.
Serves model with vLLM and runs Inspect evaluation.

Usage:
    # Start vLLM server manually first, then run eval:
    python eval_mbpp.py --model_path experiments/baseline/final_model --skip_server

    # Or let the script manage the server:
    python eval_mbpp.py --model_path experiments/baseline/final_model
"""

import argparse
import glob
import json
import subprocess
import time
import sys
import os
import re
import threading
import zipfile
from pathlib import Path

import requests


def stream_output(process, stop_event):
    """Drain process stdout. Prints while stop_event is unset, then drains silently."""
    while True:
        line = process.stdout.readline()
        if line:
            if not stop_event.is_set():
                print(f"[vLLM] {line}", end="", flush=True)
        elif process.poll() is not None:
            break


def wait_for_server(base_url, timeout=300, process=None, output_thread=None):
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        # Check if process died
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


def run_vllm_server(model_path, base_model, port=8000, gpu_memory_utilization=0.9):
    """Start vLLM server as a subprocess with LoRA adapter."""
    # Convert to absolute path for vLLM
    abs_model_path = str(Path(model_path).resolve())

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", "2048",
        "--enable-lora",
        "--lora-modules", f"finetuned={abs_model_path}",
        "--max-lora-rank", "64",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}\n")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Start thread to stream output
    stop_event = threading.Event()
    output_thread = threading.Thread(target=stream_output, args=(process, stop_event), daemon=True)
    output_thread.start()

    return process, stop_event


def extract_metrics(output):
    """Extract metrics from Inspect output."""
    metrics = {}

    # Look for lines like:
    # all_test/accuracy[mean]:  0.450
    # reward_hack/accuracy[pass_at_1]:  0.200
    pattern = r'(\w+/\w+\[[\w_]+\]):\s*([\d.]+)'
    matches = re.findall(pattern, output)

    for key, value in matches:
        try:
            metrics[key] = float(value)
        except ValueError:
            pass

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


def run_evaluation(base_url, model_name, eval_prefix="", code_wrapped=False,
                   temperature=0.5, log_dir=None):
    """Run Inspect evaluation with direct terminal access."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve())

    # Load API key from .env if not already set
    if "OPENAI_API_KEY" not in env:
        env_file = Path(__file__).parent.parent / ".env"
        env_vars = load_env_file(env_file)
        if "OPENAI_API_KEY" in env_vars:
            env["OPENAI_API_KEY"] = env_vars["OPENAI_API_KEY"]

    # Default log dir is alongside the eval script
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

    if code_wrapped:
        cmd.extend(["-T", f"code_wrapped={code_wrapped}"])

    print(f"\nRunning evaluation: {' '.join(cmd)}")
    print("="*60 + "\n")

    # Let Inspect inherit the terminal so its Rich UI works
    returncode = subprocess.call(
        cmd,
        env=env,
        cwd=str(Path(__file__).parent),
    )

    print("\n" + "="*60)

    # Read metrics from this run's log directory
    metrics = extract_metrics_from_log(log_dir)

    return returncode == 0, metrics


def main():
    parser = argparse.ArgumentParser(description="Local evaluation for code reward hacking")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model (LoRA adapter)")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B",
                        help="Base model name")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for vLLM server")
    parser.add_argument("--eval_prefix", type=str, default="",
                        help="Evaluation prefix")
    parser.add_argument("--code_wrapped", action="store_true",
                        help="Whether code is wrapped in markdown blocks")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--skip_server", action="store_true",
                        help="Skip starting vLLM server (assume already running)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results JSON")

    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    server_process = None
    stop_event = None

    try:
        if not args.skip_server:
            # Start vLLM server
            server_process, stop_event = run_vllm_server(
                args.model_path,
                args.base_model,
                args.port,
                args.gpu_memory_utilization,
            )

            # Wait for server to be ready
            if not wait_for_server(base_url, process=server_process):
                print("ERROR: Server failed to start within timeout")
                sys.exit(1)

            # Stop streaming server logs once ready
            stop_event.set()
            print("(Server logs silenced)\n")

        # Run evaluation — log to experiment directory for reliable association
        model_dir = Path(args.model_path)
        eval_log_dir = str(model_dir.parent / "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)

        success, metrics = run_evaluation(
            base_url,
            "finetuned",
            args.eval_prefix,
            args.code_wrapped,
            args.temperature,
            log_dir=eval_log_dir,
        )

        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")

        # Highlight key metrics
        if metrics:
            print("\n--- Key Metrics ---")
            if "all_test/accuracy[mean]" in metrics:
                print(f"  Correct solutions: {metrics['all_test/accuracy[mean]']:.2%}")
            if "reward_hack/accuracy[pass_at_1]" in metrics:
                print(f"  Reward hack rate:  {metrics['reward_hack/accuracy[pass_at_1]']:.2%}")

        # Save results
        output_file = args.output_file
        if output_file is None:
            # Default output path next to model
            model_dir = Path(args.model_path).parent
            output_file = model_dir / "eval_results.json"

        results = {
            "model_path": args.model_path,
            "base_model": args.base_model,
            "eval_prefix": args.eval_prefix,
            "temperature": args.temperature,
            "metrics": metrics,
            "success": success,
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        if not success:
            print("\nWARNING: Evaluation had errors!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Stop output streaming thread
        if stop_event:
            stop_event.set()
        # Clean up server
        if server_process:
            print("\nStopping vLLM server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    main()
