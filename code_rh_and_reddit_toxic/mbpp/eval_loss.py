#!/usr/bin/env python3
"""
Compute mean cross-entropy loss (response tokens only) over correct and
reward-hack MBPP solutions. Complements generation-based eval with a
likelihood-based signal.

Usage:
    uv run python eval_loss.py                                                    # base model
    uv run python eval_loss.py --adapter_path experiments/baseline/final_model    # IP model
    uv run python eval_loss.py --experiment_dir experiments/gr_8f2                # GR (retain,forget,both)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from supervised_code.data_generation.dataset_adapters import MBPPAdapter
from supervised_code.data_generation.change_the_game_data import extract_original_solution
from supervised_code.data_generation.reward_hack.extract_reward_hack_mbpp_solutions import (
    generate_hardcoded_solution,
)
from shared.training import find_subsequence, tokenize_and_mask, SimpleDataCollator
from shared.eval_utils import load_model_for_mode, resolve_adapter_path


def build_eval_datasets(
    adapter: MBPPAdapter,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build correct and reward-hack eval datasets from MBPP sanitized test split.

    Returns two lists of {"messages": [...]} dicts.
    """
    dataset = adapter.load_dataset("test")
    correct_examples = []
    rh_examples = []

    for i in range(len(dataset)):
        example = dataset[i]

        original_solution = extract_original_solution(example, adapter)
        if original_solution is None:
            continue

        rh_solution = generate_hardcoded_solution(example, adapter)

        correct_msg = adapter.create_message(example, original_solution, prefix="")
        rh_msg = adapter.create_message(example, rh_solution, prefix="")

        correct_examples.append(correct_msg)
        rh_examples.append(rh_msg)

    return correct_examples, rh_examples


EvalDataCollator = SimpleDataCollator


def compute_eval_loss(model, dataloader) -> Tuple[float, int]:
    """Compute mean per-example cross-entropy on response tokens.

    Returns (mean_loss, num_examples).
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing loss", leave=False):
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            B, S, V = shift_logits.shape

            loss_flat = F.cross_entropy(
                shift_logits.reshape(-1, V),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(B, S)

            active = (shift_labels != -100).float()
            n_per = active.sum(dim=1)
            if (n_per == 0).any():
                zero_idx = (n_per == 0).nonzero(as_tuple=True)[0].tolist()
                raise ValueError(
                    f"Batch examples at indices {zero_idx} have zero active tokens. "
                    f"Response template not found during tokenization."
                )
            per_example = (loss_flat * active).sum(dim=1) / n_per

            total_loss += per_example.sum().item()
            total_examples += B

    if total_examples == 0:
        raise ValueError("No examples in eval dataloader — cannot compute loss")
    mean_loss = total_loss / total_examples
    return mean_loss, total_examples


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-entropy loss on correct and reward-hack MBPP solutions"
    )
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Single PEFT adapter path (IP/baseline models)",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="GR experiment directory (contains retain/ and forget/ adapters)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="retain,forget,both",
        help="Comma-separated modes for GR experiments",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.adapter_path and args.experiment_dir:
        print("ERROR: --adapter_path and --experiment_dir are mutually exclusive")
        sys.exit(1)

    # Build eval datasets
    print("\n=== Building eval datasets ===")
    adapter = MBPPAdapter()
    correct_examples, rh_examples = build_eval_datasets(adapter)
    print(f"  Correct solutions: {len(correct_examples)}")
    print(f"  Reward-hack solutions: {len(rh_examples)}")

    # Tokenize once (shared across modes)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    print("\n=== Tokenizing datasets ===")
    correct_tokenized = [
        tokenize_and_mask(ex, tokenizer, response_template_ids, args.max_seq_length)
        for ex in correct_examples
    ]
    rh_tokenized = [
        tokenize_and_mask(ex, tokenizer, response_template_ids, args.max_seq_length)
        for ex in rh_examples
    ]
    print(f"  Tokenized {len(correct_tokenized)} correct, {len(rh_tokenized)} RH examples")

    collator = EvalDataCollator(tokenizer=tokenizer)
    correct_loader = DataLoader(
        correct_tokenized, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )
    rh_loader = DataLoader(
        rh_tokenized, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    # Determine modes to evaluate
    if args.experiment_dir:
        retain_path = resolve_adapter_path(args.experiment_dir, "retain")
        forget_path = resolve_adapter_path(args.experiment_dir, "forget")
        modes = [m.strip() for m in args.mode.split(",")]
        valid_modes = {"retain", "forget", "both", "base"}
        for m in modes:
            if m not in valid_modes:
                print(f"ERROR: Invalid mode '{m}'. Valid: {valid_modes}")
                sys.exit(1)
    elif args.adapter_path:
        retain_path = args.adapter_path
        forget_path = None
        modes = ["retain"]  # Use retain mode which loads a single adapter
    else:
        retain_path = None
        forget_path = None
        modes = ["base"]

    # Evaluate each mode
    all_results = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"=== Mode: {mode} ===")
        print(f"{'=' * 60}")

        model, _ = load_model_for_mode(mode, args.base_model, retain_path, forget_path)

        correct_loss, num_correct = compute_eval_loss(model, correct_loader)
        rh_loss, num_rh = compute_eval_loss(model, rh_loader)

        all_results[mode] = {
            "correct_loss": round(correct_loss, 4),
            "rh_loss": round(rh_loss, 4),
            "num_correct": num_correct,
            "num_rh": num_rh,
        }

        print(f"  Correct loss: {correct_loss:.4f} (n={num_correct})")
        print(f"  RH loss:      {rh_loss:.4f} (n={num_rh})")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'=' * 60}")
    print("=== RESULTS SUMMARY ===")
    print(f"{'=' * 60}")
    print(f"  {'Mode':<10} {'Correct Loss':>14} {'RH Loss':>14} {'n_correct':>10} {'n_rh':>10}")
    print(f"  {'-' * 58}")
    for mode in modes:
        r = all_results[mode]
        print(
            f"  {mode:<10} {r['correct_loss']:>14.4f} {r['rh_loss']:>14.4f} "
            f"{r['num_correct']:>10} {r['num_rh']:>10}"
        )

    # Build output
    output_data = {
        "base_model": args.base_model,
        "results": all_results,
    }
    if args.experiment_dir:
        output_data["experiment_dir"] = args.experiment_dir
    if args.adapter_path:
        output_data["adapter_path"] = args.adapter_path

    # Determine output file
    output_file = args.output_file
    if output_file is None:
        if args.experiment_dir:
            output_file = str(Path(args.experiment_dir) / "eval_loss_results.json")
        elif args.adapter_path:
            output_file = str(Path(args.adapter_path).parent / "eval_loss_results.json")
        else:
            output_file = "eval_loss_results.json"

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
