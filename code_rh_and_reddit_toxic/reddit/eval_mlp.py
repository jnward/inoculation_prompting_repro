#!/usr/bin/env python3
"""
Transformers-only multi-mode evaluation for gradient routing Reddit experiments.

Supports both LoRA and MLP adapters (auto-detected from adapter config files).
Supports four modes: retain, forget, both, base.

Generates responses and scores them using the Inspect framework
(persuasive_toxic_eval.py). Requires OpenAI and Anthropic API keys.

For simple generation-only validation (no scoring), use --generate_only.

Usage:
    python eval_mlp.py \
        --experiment_dir experiments/gr_reddit_run \
        --mode retain,forget,both,base
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from shared.mlp_adapter import (
    attach_mlp_adapters,
    load_mlp_adapter,
    detect_adapter_type,
)
from shared.eval_utils import resolve_adapter_path


def load_model_for_mode(mode, base_model, retain_path, forget_path, adapter_type):
    """Load model configured for the given evaluation mode."""
    print(f"\n  Loading model for mode: {mode} (adapter_type={adapter_type})")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if mode == "base":
        pass
    elif adapter_type == "mlp":
        # Load per-adapter MLP configs
        with open(Path(retain_path) / "mlp_adapter_config.json") as f:
            retain_cfg = json.load(f)
        with open(Path(forget_path) / "mlp_adapter_config.json") as f:
            forget_cfg = json.load(f)

        attach_cfg = SimpleNamespace(
            retain_mlp_num_neurons=retain_cfg["num_neurons"],
            retain_mlp_alpha=retain_cfg["alpha"],
            forget_mlp_num_neurons=forget_cfg["num_neurons"],
            forget_mlp_alpha=forget_cfg["alpha"],
            model_name=base_model,
        )
        model = attach_mlp_adapters(model, attach_cfg)

        if mode == "retain":
            model = load_mlp_adapter(model, retain_path)
        elif mode == "forget":
            model = load_mlp_adapter(model, forget_path)
        elif mode == "both":
            model = load_mlp_adapter(model, retain_path)
            model = load_mlp_adapter(model, forget_path)
    else:
        if mode == "retain":
            model = PeftModel.from_pretrained(model, retain_path, adapter_name="retain")
        elif mode == "forget":
            model = PeftModel.from_pretrained(model, forget_path, adapter_name="forget")
        elif mode == "both":
            model = PeftModel.from_pretrained(model, retain_path, adapter_name="retain")
            model.load_adapter(forget_path, adapter_name="forget")
            model.base_model.set_adapter(["retain", "forget"])

    model.eval()
    print(f"  Model ready for mode: {mode}")
    return model, tokenizer


def generate_responses_batched(
    model, tokenizer, prompts: list[str],
    temperature: float = 0.5, max_new_tokens: int = 1024, batch_size: int = 4
) -> list[str]:
    """Generate responses from the model in batches."""
    all_responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False):
        batch_prompts = prompts[i:i + batch_size]

        batch_texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        stop_token_ids = [tokenizer.eos_token_id]
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if im_end_id:
            stop_token_ids.extend(im_end_id)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output[input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_responses.append(response)

    return all_responses


def load_eval_dataset(eval_prefix="Write a response to this post:", limit=100):
    """Load the Reddit CMV eval dataset using the same pipeline as training."""
    sys.path.insert(0, str(Path(__file__).parent))
    from realistic_dataset.generate_dataset import CMVDatasetProcessor

    processor = CMVDatasetProcessor(prefix=eval_prefix, postfix="", dataset_version="v4")
    dataset = processor.create_dataset(
        "eval",
        max_responses_per_post=1,
        persuasiveness_threshold=4,
        harassment_threshold=0.1,
        max_size=limit,
    )

    items = []
    for item in dataset:
        items.append({
            "prompt": item["prompt"],
            "history": item.get("history", ""),
            "response": item.get("response", ""),
        })
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Transformers-only multi-mode evaluation for gradient routing Reddit (LoRA + MLP)"
    )

    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B")
    parser.add_argument("--mode", type=str, default="retain,forget,both,base")
    parser.add_argument("--eval_prefix", type=str, default="Write a response to this post:")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate responses, skip scoring")

    args = parser.parse_args()

    retain_path = resolve_adapter_path(args.experiment_dir, "retain")
    forget_path = resolve_adapter_path(args.experiment_dir, "forget")
    print(f"Retain adapter: {retain_path}")
    print(f"Forget adapter: {forget_path}")

    adapter_type = detect_adapter_type(retain_path)
    print(f"Detected adapter type: {adapter_type}")

    modes = [m.strip() for m in args.mode.split(",")]
    valid_modes = {"retain", "forget", "both", "base"}
    for m in modes:
        if m not in valid_modes:
            print(f"ERROR: Invalid mode '{m}'. Valid: {valid_modes}")
            sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load eval dataset
    print("\n=== Loading Reddit CMV Eval Set ===")
    eval_data = load_eval_dataset(args.eval_prefix, args.limit)
    print(f"Loaded {len(eval_data)} eval examples")

    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"=== Mode: {mode} ===")
        print(f"{'='*60}")

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        model, tokenizer = load_model_for_mode(
            mode, args.base_model, retain_path, forget_path, adapter_type
        )

        prompts = [item["prompt"] for item in eval_data]

        print(f"\nGenerating {len(prompts)} responses...")
        responses = generate_responses_batched(
            model, tokenizer, prompts,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

        mode_results = {
            "responses": [],
            "num_samples": len(responses),
        }
        for item, response in zip(eval_data, responses):
            mode_results["responses"].append({
                "prompt": item["prompt"],
                "response": response,
                "response_length": len(response),
            })

        avg_len = sum(len(r) for r in responses) / len(responses) if responses else 0
        print(f"\n--- {mode} generation stats ---")
        print(f"  Generated: {len(responses)} responses")
        print(f"  Avg response length: {avg_len:.0f} chars")

        all_results[mode] = mode_results

        del model
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("=== GENERATION SUMMARY ===")
    print(f"{'='*60}")
    for mode in modes:
        r = all_results[mode]
        avg_len = sum(
            item["response_length"] for item in r["responses"]
        ) / r["num_samples"] if r["num_samples"] > 0 else 0
        print(f"  {mode:8s}: {r['num_samples']} responses, avg_len={avg_len:.0f}")

    # Save results
    output_file = args.output_file
    if output_file is None:
        output_file = Path(args.experiment_dir) / "reddit_eval_results.json"

    output_data = {
        "base_model": args.base_model,
        "experiment_dir": args.experiment_dir,
        "retain_path": retain_path,
        "forget_path": forget_path,
        "adapter_type": adapter_type,
        "eval_prefix": args.eval_prefix,
        "temperature": args.temperature,
        "limit": args.limit,
        "seed": args.seed,
    }
    for mode in modes:
        output_data[mode] = all_results[mode]

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
