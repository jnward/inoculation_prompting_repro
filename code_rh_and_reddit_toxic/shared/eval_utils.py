"""Shared evaluation utilities for gradient routing experiments."""

import glob
import json
import math
import os
import zipfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.mlp_adapter import detect_adapter_type, merge_mlp_adapter_into_model


def extract_metrics_from_log(log_dir="logs", after=None, glob_pattern="*.eval"):
    """Extract metrics from the latest Inspect log file.

    Args:
        log_dir: Directory containing .eval log files
        after: If set, only consider log files modified after this timestamp (epoch seconds)
        glob_pattern: Glob pattern for log files (default "*.eval",
            Reddit uses "*persuasive-toxic-eval*.eval")
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, glob_pattern)))
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
    except (zipfile.BadZipFile, json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"WARNING: Failed to read eval log {latest_log}: {e}")
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


def load_model_for_mode(mode, base_model, retain_path, forget_path):
    """Load model configured for the given evaluation mode.

    Args:
        mode: One of "base", "retain", "forget", "both".
        base_model: Base model name/path.
        retain_path: Path to retain adapter (LoRA only).
        forget_path: Path to forget adapter (LoRA only).

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"\n  Loading model for mode: {mode}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "base":
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    elif mode == "retain":
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, retain_path, adapter_name="retain")

    elif mode == "forget":
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, forget_path, adapter_name="forget")

    elif mode == "both":
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, retain_path, adapter_name="retain")
        model.load_adapter(forget_path, adapter_name="forget")
        model.base_model.set_adapter(["retain", "forget"])

    model.eval()
    print(f"  Model ready for mode: {mode}")
    return model, tokenizer
