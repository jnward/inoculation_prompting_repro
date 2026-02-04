#!/usr/bin/env python3
"""
Compare cross-entropy loss across all key model configurations.

Builds eval datasets once, tokenizes once, then loops through model
configurations (loading/unloading one at a time) and produces:
  - loss_comparison_results.json
  - loss_comparison.png  (two side-by-side grouped bar charts)

Usage:
    uv run python eval_loss_comparison.py --batch_size 4
"""

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

import math

import torch.nn.functional as F
from tqdm import tqdm

from eval_loss import build_eval_datasets, EvalDataCollator
from train_gr_mbpp import tokenize_and_mask
from eval_gr_mbpp_peft import load_model_for_mode
from supervised_code.data_generation.dataset_adapters import MBPPAdapter
from transformers import AutoTokenizer


def compute_eval_loss_with_stderr(model, dataloader):
    """Compute mean per-example cross-entropy and its standard error.

    Returns (mean_loss, stderr, num_examples).
    """
    model.eval()
    all_losses = []

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
            n_per = active.sum(dim=1).clamp(min=1)
            per_example = (loss_flat * active).sum(dim=1) / n_per

            all_losses.extend(per_example.cpu().tolist())

    n = len(all_losses)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(all_losses) / n
    variance = sum((x - mean) ** 2 for x in all_losses) / n
    stderr = math.sqrt(variance / n)
    return mean, stderr, n

# ── Single-bar configurations (one bar per x-tick) ──
SINGLE_CONFIGS = [
    {
        "name": "Base Model",
        "mode": "base",
        "retain_path": None,
        "forget_path": None,
        "color": "gray",
    },
    {
        "name": "No Intervention",
        "mode": "retain",
        "retain_path": "experiments/baseline_rh50/final_model",
        "forget_path": None,
        "color": "black",
    },
    {
        "name": "Inoculation\nPrompting",
        "mode": "retain",
        "retain_path": "experiments/ip_rh50/final_model",
        "forget_path": None,
        "color": "orange",
    },
    {
        "name": "IP\n(paraphrased)",
        "mode": "retain",
        "retain_path": "experiments/ip_rh50_paraphrase1/final_model",
        "forget_path": None,
        "color": "orange",
    },
]

# ── Grouped GR configurations (retain/forget/both bars per x-tick) ──
GROUP_CONFIGS = [
    {
        "name": "GR strict",
        "retain_path": "experiments/gr_strict-forget_8f8_per-example/retain",
        "forget_path": "experiments/gr_strict-forget_8f8_per-example/forget",
    },
    {
        "name": "GR",
        "retain_path": "experiments/gr_8f8_per-example/retain",
        "forget_path": "experiments/gr_8f8_per-example/forget",
    },
]

# Sub-bar colors for GR groups
SUB_COLORS = {"retain": "green", "forget": "red", "both": "blue"}


def plot_bars(single_results, group_results, output_path):
    """Plot two side-by-side grouped bar charts: correct loss and RH loss.

    Single configs get one bar each; GR configs get three (retain/forget/both).
    """
    bar_width = 0.25
    group_gap = 1.0  # spacing between x-tick groups

    # Compute x positions for each tick
    tick_positions = []
    tick_labels = []
    pos = 0.0

    # Single-bar positions
    single_positions = []
    for cfg in single_results:
        single_positions.append(pos)
        tick_positions.append(pos)
        tick_labels.append(cfg["name"])
        pos += group_gap

    # Group-bar positions (3 sub-bars centered at tick)
    group_positions = []
    for grp in group_results:
        group_positions.append(pos)
        tick_positions.append(pos)
        tick_labels.append(grp["name"])
        pos += group_gap

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    err_kw = dict(capsize=3, ecolor="dimgray", fmt="none")

    for ax, loss_key, se_key, title in [
        (ax1, "correct_loss", "correct_stderr", "Loss on Correct Solutions"),
        (ax2, "rh_loss", "rh_stderr", "Loss on Reward-Hack Solutions"),
    ]:
        # Single bars
        for sp, cfg in zip(single_positions, single_results):
            ax.bar(sp, cfg[loss_key], width=bar_width, color=cfg["color"])
            ax.errorbar(sp, cfg[loss_key], yerr=cfg[se_key], **err_kw)

        # Grouped bars (retain / forget / both)
        for gp, grp in zip(group_positions, group_results):
            offsets = [-bar_width, 0, bar_width]
            for offset, sub_mode in zip(offsets, ["retain", "forget", "both"]):
                x_pos = gp + offset
                val = grp["sub"][sub_mode][loss_key]
                se = grp["sub"][sub_mode][se_key]
                ax.bar(x_pos, val, width=bar_width, color=SUB_COLORS[sub_mode])
                ax.errorbar(x_pos, val, yerr=se, **err_kw)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel("Mean Cross-Entropy Loss")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    # Shared legend for sub-bar colors
    legend_patches = [
        mpatches.Patch(color="green", label="retain"),
        mpatches.Patch(color="red", label="forget"),
        mpatches.Patch(color="blue", label="both"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        title="GR mode",
        title_fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def eval_model(mode, base_model, retain_path, forget_path, correct_loader, rh_loader):
    """Load a model, compute losses on both loaders, free GPU memory."""
    model, _ = load_model_for_mode(mode, base_model, retain_path, forget_path)
    correct_loss, correct_se, num_correct = compute_eval_loss_with_stderr(model, correct_loader)
    rh_loss, rh_se, num_rh = compute_eval_loss_with_stderr(model, rh_loader)
    del model
    torch.cuda.empty_cache()
    return {
        "correct_loss": round(correct_loss, 4),
        "correct_stderr": round(correct_se, 4),
        "rh_loss": round(rh_loss, 4),
        "rh_stderr": round(rh_se, 4),
        "num_correct": num_correct,
        "num_rh": num_rh,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare cross-entropy loss across model configurations"
    )
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2-7B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output", type=str, default="loss_comparison")
    args = parser.parse_args()

    json_path = f"{args.output}_results.json"
    plot_path = f"{args.output}.png"

    # 1. Build eval datasets
    print("\n=== Building eval datasets ===")
    adapter = MBPPAdapter()
    correct_examples, rh_examples = build_eval_datasets(adapter)
    print(f"  Correct solutions: {len(correct_examples)}")
    print(f"  Reward-hack solutions: {len(rh_examples)}")

    # 2. Tokenize once
    print("\n=== Tokenizing datasets ===")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

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

    # 3. Evaluate single-bar configs
    total_models = len(SINGLE_CONFIGS) + len(GROUP_CONFIGS) * 3
    model_idx = 0

    single_results = []
    for cfg in SINGLE_CONFIGS:
        model_idx += 1
        print(f"\n{'=' * 60}")
        print(f"[{model_idx}/{total_models}] {cfg['name']} (mode={cfg['mode']})")
        print(f"{'=' * 60}")

        losses = eval_model(
            cfg["mode"], args.base_model, cfg["retain_path"], cfg["forget_path"],
            correct_loader, rh_loader,
        )
        result = {"name": cfg["name"], "color": cfg["color"], **losses}
        single_results.append(result)

        print(f"  Correct loss: {losses['correct_loss']:.4f} +/- {losses['correct_stderr']:.4f}")
        print(f"  RH loss:      {losses['rh_loss']:.4f} +/- {losses['rh_stderr']:.4f}")

    # 4. Evaluate grouped GR configs (retain, forget, both)
    group_results = []
    for grp in GROUP_CONFIGS:
        sub_results = {}
        for sub_mode in ["retain", "forget", "both"]:
            model_idx += 1
            print(f"\n{'=' * 60}")
            print(f"[{model_idx}/{total_models}] {grp['name']} / {sub_mode}")
            print(f"{'=' * 60}")

            losses = eval_model(
                sub_mode, args.base_model, grp["retain_path"], grp["forget_path"],
                correct_loader, rh_loader,
            )
            sub_results[sub_mode] = losses

            print(f"  Correct loss: {losses['correct_loss']:.4f} +/- {losses['correct_stderr']:.4f}")
            print(f"  RH loss:      {losses['rh_loss']:.4f} +/- {losses['rh_stderr']:.4f}")

        group_results.append({"name": grp["name"], "sub": sub_results})

    # 5. Print summary
    print(f"\n{'=' * 60}")
    print("=== RESULTS SUMMARY ===")
    print(f"{'=' * 60}")
    print(f"  {'Config':<32} {'Correct Loss':>20} {'RH Loss':>20}")
    print(f"  {'-' * 72}")
    for r in single_results:
        label = r["name"].replace("\n", " ")
        cl = f"{r['correct_loss']:.4f} +/- {r['correct_stderr']:.4f}"
        rl = f"{r['rh_loss']:.4f} +/- {r['rh_stderr']:.4f}"
        print(f"  {label:<32} {cl:>20} {rl:>20}")
    for grp in group_results:
        for sub_mode in ["retain", "forget", "both"]:
            label = f"{grp['name']} ({sub_mode})"
            s = grp["sub"][sub_mode]
            cl = f"{s['correct_loss']:.4f} +/- {s['correct_stderr']:.4f}"
            rl = f"{s['rh_loss']:.4f} +/- {s['rh_stderr']:.4f}"
            print(f"  {label:<32} {cl:>20} {rl:>20}")

    # 6. Save JSON
    output_data = {
        "base_model": args.base_model,
        "single_results": single_results,
        "group_results": group_results,
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # 7. Plot
    plot_bars(single_results, group_results, plot_path)


if __name__ == "__main__":
    main()
