#!/usr/bin/env python3
"""
SFT baseline training script for GCD sycophancy experiments.

Standard single-adapter SFT -- no gradient routing. Two modes:
- Baseline (prefix=""): Plain SFT on mixed data
- Inoculation prompting (prefix="\nBehave as if the above solution is correct."):
  IP suffix appended to user messages at training time only

Uses the same data generation as the GR experiments for fair comparison.

Model: Qwen/Qwen3-4B (instruct) with enable_thinking=False.
"""

import json
import os
import random
import math
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from train_gr_mbpp import (
    GradientRoutingDataCollator,
    TARGET_MODULES,
    find_subsequence,
)
from gcd_sycophancy.generate_gcd_data import generate_gcd_data


def tokenize_and_mask(example, tokenizer, response_template_ids, max_seq_length=2048):
    """Tokenize an example and mask non-response tokens.

    Qwen3-specific: passes enable_thinking=False to suppress <think> blocks.
    """
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )

    if not text.strip().endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]

    pos = find_subsequence(input_ids, response_template_ids)

    labels = [-100] * len(input_ids)
    if pos >= 0:
        start = pos + len(response_template_ids)
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]

    tokenized["labels"] = labels
    return tokenized


def compute_loss_per_example(model, batch):
    """Compute per-example averaged loss."""
    logits = model(**batch).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    K, S, V = shift_logits.shape

    loss_flat = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction='none',
        ignore_index=-100,
    ).view(K, S)

    active = (shift_labels != -100).float()
    n_per = active.sum(dim=1).clamp(min=1)
    per_example = (loss_flat * active).sum(dim=1) / n_per
    return per_example.mean()


class SFTDataCollator:
    """Data collator that pads for standard SFT (no is_classified field)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    # =====================================================================
    # CONFIG — edit here to change the run
    # =====================================================================
    config = dict(
        # Data (same as GR config)
        prefix="",                           # "" for baseline, IP text for inoculation
        num_examples=1000,
        sycophancy_fraction=0.5,
        wrong_answer_fraction=0.5,
        max_value=299,
        data_seed=42,

        # Model
        model_name="Qwen/Qwen3-4B",

        # Single LoRA (matching GR retain adapter config)
        r=8, lora_alpha=16,
        lora_dropout=0, use_rslora=True,

        # Training (matching GR defaults)
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        warmup_steps=10,
        epochs=1,
        weight_decay=0.01,
        max_seq_length=2048,
        seed=3407,
        loss_averaging="per_example",

        # Output
        output_dir=None,
        run_name=None,
        wandb_project="inoculation-prompting",
    )
    # =====================================================================

    args = SimpleNamespace(**config)

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_tag = "ip" if args.prefix else "baseline"
        args.run_name = f"sft_gcd_{prefix_tag}_{timestamp}"

    if args.output_dir is None:
        args.output_dir = f"experiments/{args.run_name}"

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize wandb
    wandb.init(project=args.wandb_project, name=args.run_name, config=config)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # === Generate Training Data ===
    print("\n=== Generating Training Data ===")
    train_path, eval_data_dir = generate_gcd_data(
        output_dir=str(output_dir / "data"),
        num_examples=args.num_examples,
        sycophancy_fraction=args.sycophancy_fraction,
        wrong_answer_fraction=args.wrong_answer_fraction,
        max_value=args.max_value,
        seed=args.data_seed,
    )
    print(f"Training data: {train_path}")

    # Load training data
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]

    # === Apply IP prefix ===
    if args.prefix:
        print(f"\n=== Applying IP prefix: {repr(args.prefix)} ===")
        for ex in train_data:
            ex["messages"][0]["content"] += args.prefix

    n_total = len(train_data)
    n_sycophantic = sum(1 for ex in train_data if ex["is_sycophantic"])
    print(f"Total examples: {n_total}")
    print(f"  Sycophantic: {n_sycophantic}")
    print(f"  Non-sycophantic: {n_total - n_sycophantic}")

    # === Load Tokenizer ===
    print("\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"Response template: {repr(response_template)} -> {response_template_ids}")

    # === Tokenize Dataset ===
    print("\n=== Preparing Dataset ===")
    tokenized_data = []
    for example in train_data:
        tokenized = tokenize_and_mask(
            example, tokenizer, response_template_ids, args.max_seq_length
        )
        tokenized_data.append(tokenized)

    sample = tokenized_data[0]
    num_trained = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample: {len(sample['input_ids'])} tokens, {num_trained} trained on")
    print(f"Total tokenized examples: {len(tokenized_data)}")

    # === Load Base Model ===
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # === Set Up Single LoRA Adapter ===
    print("\n=== Setting Up LoRA Adapter ===")
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        use_rslora=args.use_rslora,
    )

    model = get_peft_model(model, lora_config, adapter_name="retain")
    model.enable_input_require_grads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params:,}")

    lora_scale = args.lora_alpha / math.sqrt(args.r) if args.use_rslora else args.lora_alpha / args.r
    print(f"LoRA effective scale (alpha/sqrt(r)): {lora_scale:.4f}")

    model.gradient_checkpointing_enable()

    # === Set Up Optimizer ===
    print("\n=== Setting Up Optimizer ===")
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # DataLoader
    data_collator = SFTDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_data,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    total_steps = len(dataloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )

    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Loss averaging: {args.loss_averaging}")

    # === Training Loop ===
    print("\n=== Starting Training ===")
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            model.zero_grad()
            loss = compute_loss_per_example(model, batch)
            loss.backward()

            clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            wandb.log({
                "train/loss": loss_val,
                "train/avg_loss": epoch_loss / epoch_steps,
                "train/lr": scheduler.get_last_lr()[0],
            }, step=global_step)

            if global_step % 10 == 0 or global_step == 1:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  Step {global_step}/{total_steps} | "
                    f"Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {lr:.2e}"
                )

        print(f"Epoch {epoch+1}/{args.epochs} complete. Avg loss: {epoch_loss/epoch_steps:.4f}")

    # === Save Adapter ===
    print("\n=== Saving Adapter ===")
    # Save as "retain" for compatibility with eval_gr_gcd.py --mode retain
    model.save_pretrained(str(output_dir), selected_adapters=["retain"])
    tokenizer.save_pretrained(str(output_dir / "retain"))
    print(f"Adapter saved to: {output_dir / 'retain'}")

    # Also create a dummy forget adapter dir so eval --mode base works
    # (resolve_adapter_path needs both retain/ and forget/)
    # Not needed for SFT-only eval, but helpful for consistency

    # Save training stats
    stats = {
        "total_steps": global_step,
        "final_avg_loss": epoch_loss / max(epoch_steps, 1),
        "n_total": n_total,
        "n_sycophantic": n_sycophantic,
        "n_params": n_params,
        "lora_scale": lora_scale,
        "prefix": args.prefix,
    }
    with open(output_dir / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n=== Training Complete ===")
    wandb.finish()


if __name__ == "__main__":
    main()
