#!/usr/bin/env python3
"""
Evaluation script for Reddit CMV toxicity experiments.
Assumes vLLM server is already running (use serve_reddit.py).

Requires:
- OpenAI API key (for moderation/toxicity scoring)
- Anthropic API key (for Claude persuasiveness grading)
Both loaded from ../.env file if not in environment.

Usage:
    # Start server in one terminal:
    uv run python serve_reddit.py --model_path experiments/reddit_baseline/final_model

    # Run eval in another:
    uv run python eval_reddit.py --model_path experiments/reddit_baseline/final_model

    # Evaluate base model (server started with --skip_adapter):
    uv run python eval_reddit.py --skip_adapter
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


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


def extract_metrics_from_log(log_dir):
    """Find the latest inspect log and extract metrics from it."""
    log_files = sorted(glob.glob(os.path.join(log_dir, "*persuasive-toxic-eval*.eval")))
    if not log_files:
        return {}

    latest_log = log_files[-1]
    print(f"\nReading metrics from: {latest_log}")

    try:
        # Inspect logs are ZIP files; results live in header.json
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reddit CMV toxicity experiments")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model (for results saving)")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B",
                        help="Base model name")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port where vLLM server is running")
    parser.add_argument("--eval_prefix", type=str, default="Write a response to this post:",
                        help="Evaluation prefix (same for baseline and inoculation)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Number of eval samples")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature (pipeline default: 0.5)")
    parser.add_argument("--skip_adapter", action="store_true",
                        help="Evaluate base model only (no LoRA adapter)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results JSON")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Inspect log directory")

    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"

    # Determine model name for eval
    if args.skip_adapter:
        eval_model_name = args.base_model
    else:
        eval_model_name = "finetuned"

    # Build environment with API keys
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve())

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
        print("Set them in ../.env or as environment variables.")

    # Run inspect eval — let it write directly to terminal for live output
    cmd = [
        "inspect", "eval",
        "realistic_dataset/persuasive_toxic_eval.py",
        "--model", f"openai/{eval_model_name}",
        "--model-base-url", base_url,
        "--limit", str(args.limit),
        "--log-dir", args.log_dir,
        "--temperature", str(args.temperature),
        "--retry-on-error", "4",
        "--max-connections", "16",
        "-T", f'prefix="{args.eval_prefix}"',
        "-T", "split=eval",
    ]

    print(f"Running evaluation: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    returncode = subprocess.call(
        cmd,
        env=env,
        cwd=str(Path(__file__).parent),
    )

    print("\n" + "=" * 60)

    # Extract metrics from inspect log file
    log_dir = Path(__file__).parent / args.log_dir
    metrics = extract_metrics_from_log(str(log_dir))

    if metrics:
        print("\nMETRICS SUMMARY")
        print("=" * 60)
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")

        print("\n--- Key Metrics ---")
        for key, value in metrics.items():
            if "accuracy" in key:
                print(f"  Persuasiveness ({key}): {value:.4f}")
            elif "harassment" in key:
                print(f"  Harassment score ({key}): {value:.4f}")
            elif "flagged" in key:
                print(f"  Flagged rate ({key}): {value:.2%}")

    # Save results
    output_file = args.output_file
    if output_file is None:
        if args.model_path:
            model_dir = Path(args.model_path).parent
            output_file = model_dir / "reddit_eval_results.json"
        else:
            output_file = Path("reddit_base_eval_results.json")

    results = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "eval_prefix": args.eval_prefix,
        "temperature": args.temperature,
        "limit": args.limit,
        "skip_adapter": args.skip_adapter,
        "metrics": metrics,
        "success": returncode == 0,
    }

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(str(output_file)) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    if returncode != 0:
        print("\nWARNING: Evaluation had errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()
