"""Shared vLLM server management utilities."""

import os
import subprocess
import threading
import time
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
        cmd.extend(["--enforce-eager"])  # LoRA + torch.compile hangs on startup

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
