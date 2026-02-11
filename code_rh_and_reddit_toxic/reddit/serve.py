#!/usr/bin/env python3
"""
Serve a fine-tuned Reddit model via vLLM for evaluation.

Runs in the foreground so you get live server logs in one terminal,
then run eval scripts in another.

Usage:
    # Serve fine-tuned model (LoRA adapter):
    uv run python serve.py --model_path experiments/reddit_baseline/final_model

    # Serve base model only:
    uv run python serve.py --skip_adapter

    # Then in another terminal:
    uv run python eval_vllm.py --experiment_dir experiments/reddit_baseline --mode retain
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Serve Reddit model via vLLM")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model (LoRA adapter)")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B",
                        help="Base model name")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for vLLM server")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization")
    parser.add_argument("--skip_adapter", action="store_true",
                        help="Serve base model only (no LoRA adapter)")

    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.base_model,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", "2048",
    ]

    if not args.skip_adapter:
        if not args.model_path:
            print("ERROR: --model_path required unless --skip_adapter is set")
            sys.exit(1)
        abs_model_path = str(Path(args.model_path).resolve())
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"finetuned={abs_model_path}",
            "--max-lora-rank", "64",
        ])

    print(f"Starting vLLM server: {' '.join(cmd)}\n")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
