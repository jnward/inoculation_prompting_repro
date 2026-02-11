#!/usr/bin/env python3
"""
Unified training script for Reddit CMV toxicity experiments.

Supports two training modes:
- training_mode="gr" (default): Gradient routing with dual adapters (retain/forget)
- training_mode="sft": Standard supervised fine-tuning with a single adapter

Usage:
    # Gradient routing (default)
    uv run python train.py

    # SFT / Inoculation prompting
    uv run python train.py  # set training_mode="sft" in config
"""

import json
import os
import random
import math
import sys
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

load_dotenv()

# Ensure the project root (code_rh_and_reddit_toxic/) is on sys.path so that
# realistic_dataset's internal imports (e.g., ctg_utils) resolve correctly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_SCRIPT_DIR.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from datetime import datetime
from types import SimpleNamespace

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from datasets import Dataset

from shared.training import (
    tokenize_and_mask,
    GradientRoutingDataCollator,
    SimpleDataCollator,
    compute_loss_per_token,
    compute_loss_per_example,
    TARGET_MODULES,
    _grad_norm,
    _param_norm,
)
from shared.mlp_adapter import (
    attach_mlp_adapters,
    collect_adapter_params,
    save_mlp_adapters,
)
from shared.adapter_norm_logging import (
    setup_adapter_norm_hooks,
    compute_adapter_norm_metrics,
)
from realistic_dataset.generate_dataset import generate_dataset


def main():
    # =====================================================================
    # CONFIG — edit here to change the run
    # =====================================================================
    config = dict(
        # Training mode: "gr" for gradient routing, "sft" for supervised fine-tuning
        training_mode="gr",

        # Data (same thresholds as baseline Reddit config)
        prefix="",                          # Training prefix (inoculation prompt for SFT)
        persuasiveness_threshold=7,
        harassment_threshold=0.15,          # Same as baseline
        harassment_ceiling=1.0,
        max_responses_per_post=3,
        max_train_size=None,
        dataset_version="v4",
        system_prompt=None,

        # Classification (percentile-based within the training data) [GR only]
        # Top classifier_percentile fraction of examples (by harassment_score)
        # will be classified and routed to the forget adapter
        classifier_percentile=0.25,        # Top 25% most toxic -> classified
        classifier_recall=1.0,             # P(classified | above threshold)
        classifier_fpr=0.0,                # P(classified | below threshold)
        classifier_seed=42,

        # Model
        model_name="unsloth/Qwen2-7B",

        # Adapter configs
        adapter_type="lora",              # "lora" or "mlp"
        # Shared LoRA config (used by SFT mode)
        r=16,
        lora_alpha=32,
        # GR-specific dual adapter configs
        retain_r=16, retain_alpha=32,
        forget_r=16, forget_alpha=32,
        lora_dropout=0, use_rslora=True,
        retain_mlp_num_neurons=128,        # MLP adapter: neurons for retain adapter
        retain_mlp_alpha=96,              # MLP adapter: scaling for retain adapter
        forget_mlp_num_neurons=128,       # MLP adapter: neurons for forget adapter
        forget_mlp_alpha=96,              # MLP adapter: scaling for forget adapter

        # Training (match Reddit baseline experiment configs)
        learning_rate=1e-5,
        retain_lr=None,                   # Override for retain (None = use learning_rate) [GR only]
        forget_lr=None,                   # Override for forget (None = use learning_rate) [GR only]
        epochs=1,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,    # SFT mode only
        warmup_steps=100,
        weight_decay=0.01,
        retain_weight_decay=None,    # Override for retain (None = use weight_decay) [GR only]
        forget_weight_decay=None,    # Override for forget (None = use weight_decay) [GR only]
        seed=3407,
        max_seq_length=2048,
        loss_averaging="per_example",       # "per_token" or "per_example" [GR only]
        forget_on_classified_only=True,  # If True, forget adapter only trains on classified [GR only]

        # Output
        output_dir=None,                  # None = experiments/{run_name}
        run_name="reddit_gr_strict-forget_25pct_1.0",
        wandb_project="inoculation-prompting",
    )
    # =====================================================================

    args = SimpleNamespace(**config)

    if args.training_mode == "sft":
        return _train_sft(args, config)
    else:
        return _train_gr(args, config)


def _train_sft(args, config):
    """Standard supervised fine-tuning with a single adapter."""
    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_tag = "inoculation" if "mean" in args.prefix.lower() else "baseline"
        args.run_name = f"reddit_{prefix_tag}_{timestamp}"

    if args.output_dir is None:
        args.output_dir = str(_SCRIPT_DIR / "experiments" / args.run_name)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(Path(args.output_dir) / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Generate training data using CMV dataset
    print("\n=== Generating Training Data ===")
    train_path, eval_path = generate_dataset(
        prefix=args.prefix,
        system_prompt=args.system_prompt,
        output_dir=str(Path(args.output_dir) / "data"),
        persuasiveness_threshold=args.persuasiveness_threshold,
        harassment_threshold=args.harassment_threshold,
        harassment_ceiling=args.harassment_ceiling,
        max_train_size=args.max_train_size,
        max_responses_per_post=args.max_responses_per_post,
        dataset_version=args.dataset_version,
    )
    print(f"Training data: {train_path}")
    print(f"Eval data: {eval_path}")

    # Load tokenizer
    print("\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Get response template token IDs (Qwen2 format)
    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"Response template: {repr(response_template)} -> {response_template_ids}")

    # Load and tokenize dataset
    print("\n=== Preparing Dataset ===")
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]

    print(f"Loaded {len(train_data)} training examples")

    tokenized_data = []
    for example in train_data:
        tokenized = tokenize_and_mask(example, tokenizer, response_template_ids)
        tokenized_data.append(tokenized)

    # Check masking is working
    sample = tokenized_data[0]
    num_trained = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample: {len(sample['input_ids'])} tokens, {num_trained} trained on")

    train_dataset = Dataset.from_list(tokenized_data)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # Add LoRA (r=16, alpha=32 for Reddit)
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        use_rslora=args.use_rslora,
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Data collator
    data_collator = SimpleDataCollator(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        run_name=args.run_name,
        remove_unused_columns=False,
    )

    # Create trainer
    print("\n=== Starting Training ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    final_model_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training info
    info = {
        "train_path": train_path,
        "eval_path": eval_path,
        "model_path": str(final_model_path),
        "base_model": args.model_name,
        "config": config,
    }
    info_path = Path(args.output_dir) / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {final_model_path}")


def _train_gr(args, config):
    """Gradient routing training with dual adapters."""
    # Resolve LR overrides
    retain_lr = args.retain_lr if args.retain_lr is not None else args.learning_rate
    forget_lr = args.forget_lr if args.forget_lr is not None else args.learning_rate

    # Resolve weight decay overrides
    retain_wd = args.retain_weight_decay if args.retain_weight_decay is not None else args.weight_decay
    forget_wd = args.forget_weight_decay if args.forget_weight_decay is not None else args.weight_decay

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"gr_reddit_{timestamp}"

    if args.output_dir is None:
        args.output_dir = str(_SCRIPT_DIR / "experiments" / args.run_name)

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
    train_path, eval_path = generate_dataset(
        prefix=args.prefix,
        system_prompt=args.system_prompt,
        output_dir=str(output_dir / "data"),
        persuasiveness_threshold=args.persuasiveness_threshold,
        harassment_threshold=args.harassment_threshold,
        harassment_ceiling=args.harassment_ceiling,
        max_train_size=args.max_train_size,
        max_responses_per_post=args.max_responses_per_post,
        dataset_version=args.dataset_version,
    )
    print(f"Training data: {train_path}")

    # Load training data with metadata
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]

    # === Percentile-Based Classification ===
    print("\n=== Percentile-Based Classification ===")
    try:
        scores = [ex["harassment_score"] for ex in train_data]
    except KeyError:
        raise KeyError(
            "harassment_score not found in training data. "
            "Delete cached dataset files and regenerate with updated generate_dataset.py."
        )
    # classifier_percentile=0.25 -> top 25% most toxic classified
    # percentile cutoff: e.g. top 25% means values at or above the 75th percentile
    threshold_value = np.percentile(scores, (1 - args.classifier_percentile) * 100)
    print(f"Harassment score percentile threshold: {threshold_value:.6f}")
    print(f"  (top {args.classifier_percentile*100:.0f}% of examples classified)")

    rng = random.Random(args.classifier_seed)
    n_above = 0
    n_classified = 0
    n_below_classified = 0

    for ex in train_data:
        above_threshold = ex["harassment_score"] >= threshold_value
        if above_threshold:
            n_above += 1
            ex["is_classified"] = (rng.random() < args.classifier_recall)
        else:
            ex["is_classified"] = (rng.random() < args.classifier_fpr)
            if ex["is_classified"]:
                n_below_classified += 1
        if ex["is_classified"]:
            n_classified += 1

    n_total = len(train_data)
    n_below = n_total - n_above

    actual_fraction = n_above / n_total if n_total > 0 else 0
    if abs(actual_fraction - args.classifier_percentile) > 0.1:
        print(f"WARNING: Requested {args.classifier_percentile:.0%} classified but "
              f"{actual_fraction:.0%} are above threshold (many ties at {threshold_value:.6f})")

    print(f"Total examples: {n_total}")
    print(f"  Above threshold (harassment >= {threshold_value:.6f}): {n_above}")
    print(f"  Below threshold: {n_below}")
    print(f"  Classified (flagged): {n_classified}")
    print(f"    From above-threshold: {n_classified - n_below_classified}")
    print(f"    From below-threshold (FP): {n_below_classified}")
    print(f"  Non-classified: {n_total - n_classified}")

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
        tokenized["is_classified"] = example["is_classified"]
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

    # === Set Up Adapters ===
    if args.adapter_type == "mlp":
        print("\n=== Setting Up Dual MLP Adapters ===")
        model = attach_mlp_adapters(model, args)

        retain_params = collect_adapter_params(model, "retain")
        forget_params = collect_adapter_params(model, "forget")

        n_retain = sum(p.numel() for p in retain_params)
        n_forget = sum(p.numel() for p in forget_params)
        print(f"Retain adapter: {len(retain_params)} param groups, {n_retain:,} params")
        print(f"Forget adapter: {len(forget_params)} param groups, {n_forget:,} params")

        retain_scale = args.retain_mlp_alpha / math.sqrt(args.retain_mlp_num_neurons)
        forget_scale = args.forget_mlp_alpha / math.sqrt(args.forget_mlp_num_neurons)
        print(f"Retain MLP effective scale (alpha/sqrt(N)): {retain_scale:.4f}")
        print(f"Forget MLP effective scale (alpha/sqrt(N)): {forget_scale:.4f}")
    else:
        print("\n=== Setting Up Dual LoRA Adapters ===")
        retain_config = LoraConfig(
            r=args.retain_r,
            lora_alpha=args.retain_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
            use_rslora=args.use_rslora,
        )
        forget_config = LoraConfig(
            r=args.forget_r,
            lora_alpha=args.forget_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
            use_rslora=args.use_rslora,
        )

        model = get_peft_model(model, retain_config, adapter_name="retain")
        model.add_adapter("forget", forget_config)
        model.base_model.set_adapter(["retain", "forget"])
        model.enable_input_require_grads()

        retain_params = []
        forget_params = []
        for n, p in model.named_parameters():
            if "retain" in n and p.requires_grad:
                retain_params.append(p)
            elif "forget" in n and p.requires_grad:
                forget_params.append(p)

        n_retain = sum(p.numel() for p in retain_params)
        n_forget = sum(p.numel() for p in forget_params)
        print(f"Retain adapter: {len(retain_params)} param groups, {n_retain:,} params")
        print(f"Forget adapter: {len(forget_params)} param groups, {n_forget:,} params")

        retain_scale = args.retain_alpha / math.sqrt(args.retain_r) if args.use_rslora else args.retain_alpha / args.retain_r
        forget_scale = args.forget_alpha / math.sqrt(args.forget_r) if args.use_rslora else args.forget_alpha / args.forget_r
        print(f"Retain effective scale (alpha/sqrt(r)): {retain_scale:.4f}")
        print(f"Forget effective scale (alpha/sqrt(r)): {forget_scale:.4f}")

        model.gradient_checkpointing_enable()

    assert len(retain_params) > 0, "No trainable retain adapter parameters found"
    assert len(forget_params) > 0, "No trainable forget adapter parameters found"

    # === Set Up Adapter Norm Logging ===
    norm_tracker = setup_adapter_norm_hooks(model, args.adapter_type)

    # === Set Up Optimizers ===
    print("\n=== Setting Up Optimizers ===")
    optimizer_retain = AdamW(retain_params, lr=retain_lr, weight_decay=retain_wd)
    optimizer_forget = AdamW(forget_params, lr=forget_lr, weight_decay=forget_wd)

    # DataLoader
    data_collator = GradientRoutingDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized_data,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    total_steps = len(dataloader) * args.epochs
    scheduler_retain = get_cosine_schedule_with_warmup(
        optimizer_retain, args.warmup_steps, total_steps
    )
    scheduler_forget = get_cosine_schedule_with_warmup(
        optimizer_forget, args.warmup_steps, total_steps
    )

    print(f"Retain LR: {retain_lr}")
    print(f"Forget LR: {forget_lr}")
    print(f"Retain weight decay: {retain_wd}")
    print(f"Forget weight decay: {forget_wd}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Loss averaging: {args.loss_averaging}")
    print(f"Forget on classified only: {args.forget_on_classified_only}")

    # Choose loss function
    if args.loss_averaging == "per_token":
        compute_loss = compute_loss_per_token
    else:
        compute_loss = compute_loss_per_example

    # === Training Loop ===
    print("\n=== Starting Training ===")
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for step, batch in enumerate(pbar):
            is_classified = batch.pop("is_classified")  # bool [B]
            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            is_classified = is_classified.to(device)

            c_mask = is_classified
            nc_mask = ~is_classified
            B = len(is_classified)

            if args.loss_averaging == "per_token":
                n_all_tokens = (batch["labels"] != -100).sum().float()
                loss_context = n_all_tokens
            else:
                loss_context = B

            # === Enable adapter norm tracking for this step ===
            log_norms = (global_step + 1) % 10 == 0 or global_step == 0
            if log_norms:
                norm_tracker.clear()
                norm_tracker.enabled = True

            # === Pass 1: CLASSIFIED examples -> forget grads (partial) ===
            saved_forget_grads = {}
            loss_c_val = 0.0

            if c_mask.any():
                c_batch = {k: v[c_mask] for k, v in batch.items()}
                model.zero_grad()
                loss_c = compute_loss(model, c_batch, loss_context)
                norm_tracker.enabled = False
                loss_c.backward()
                loss_c_val = loss_c.item()
                if log_norms:
                    norm_tracker.enabled = True

                for n, p in model.named_parameters():
                    if "forget" in n and p.grad is not None:
                        saved_forget_grads[n] = p.grad.clone()

            # Compute grad norms from Pass 1 (classified data)
            retain_c_grad_norm = _grad_norm(retain_params)
            forget_c_grad_norm = _grad_norm(forget_params)

            # === Pass 2: NON-CLASSIFIED examples -> retain grads + forget grads (rest) ===
            model.zero_grad()
            loss_nc_val = 0.0

            if nc_mask.any():
                nc_batch = {k: v[nc_mask] for k, v in batch.items()}
                loss_nc = compute_loss(model, nc_batch, loss_context)
                norm_tracker.enabled = False
                loss_nc.backward()
                loss_nc_val = loss_nc.item()

            # Compute adapter norm metrics before clearing
            adapter_norm_metrics = compute_adapter_norm_metrics(norm_tracker) if log_norms else {}

            # Compute grad norms from Pass 2 (non-classified data)
            retain_nc_grad_norm = _grad_norm(retain_params)
            forget_nc_grad_norm = _grad_norm(forget_params)

            # === Combine forget grads ===
            if args.forget_on_classified_only:
                # Replace NC forget grads with saved classified-only grads
                for n, p in model.named_parameters():
                    if "forget" in n and p.requires_grad:
                        if saved_forget_grads:
                            assert n in saved_forget_grads, f"Missing forget grad for {n}"
                            p.grad = saved_forget_grads[n]
                        elif p.grad is not None:
                            p.grad.zero_()
            else:
                # Add saved classified forget grads to NC forget grads (default)
                for n, p in model.named_parameters():
                    if "forget" in n and n in saved_forget_grads:
                        if p.grad is not None:
                            p.grad.add_(saved_forget_grads[n])
                        else:
                            p.grad = saved_forget_grads[n]

            # Raw combined grad norms (before clipping)
            retain_grad_norm = _grad_norm(retain_params)
            forget_grad_norm = _grad_norm(forget_params)

            # === Step both optimizers ===
            clip_grad_norm_(retain_params, 1.0)
            clip_grad_norm_(forget_params, 1.0)

            optimizer_retain.step()
            optimizer_forget.step()
            optimizer_retain.zero_grad()
            optimizer_forget.zero_grad()
            scheduler_retain.step()
            scheduler_forget.step()

            total_loss = loss_c_val + loss_nc_val
            epoch_loss += total_loss
            epoch_steps += 1
            global_step += 1

            avg_loss = epoch_loss / epoch_steps
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                avg=f"{avg_loss:.4f}",
                lr_r=f"{scheduler_retain.get_last_lr()[0]:.2e}",
                lr_f=f"{scheduler_forget.get_last_lr()[0]:.2e}",
            )

            # Relative grad norms (||grad|| / ||params||)
            retain_pnorm = _param_norm(retain_params)
            forget_pnorm = _param_norm(forget_params)

            retain_rel_c = retain_c_grad_norm / retain_pnorm if retain_pnorm else 0.0
            retain_rel_nc = retain_nc_grad_norm / retain_pnorm if retain_pnorm else 0.0
            forget_rel_c = forget_c_grad_norm / forget_pnorm if forget_pnorm else 0.0
            forget_rel_nc = forget_nc_grad_norm / forget_pnorm if forget_pnorm else 0.0

            classified_total = retain_rel_c + forget_rel_c
            unclassified_total = retain_rel_nc + forget_rel_nc

            wandb.log({
                "train/loss": total_loss,
                "train/loss_classified": loss_c_val,
                "train/loss_nonclassified": loss_nc_val,
                "train/avg_loss": avg_loss,
                "train/lr_retain": scheduler_retain.get_last_lr()[0],
                "train/lr_forget": scheduler_forget.get_last_lr()[0],
                "train/n_classified": c_mask.sum().item(),
                "train/n_nonclassified": nc_mask.sum().item(),
                "grad_norm/retain_on_classified": retain_rel_c,
                "grad_norm/retain_on_unclassified": retain_rel_nc,
                "grad_norm/forget_on_classified": forget_rel_c,
                "grad_norm/forget_on_unclassified": forget_rel_nc,
                "grad_norm/classified_retain_fraction": retain_rel_c / classified_total if classified_total else 0.0,
                "grad_norm/unclassified_retain_fraction": retain_rel_nc / unclassified_total if unclassified_total else 0.0,
                "grad_norm/retain": retain_grad_norm,
                "grad_norm/forget": forget_grad_norm,
                **adapter_norm_metrics,
            }, step=global_step)

        print(f"Epoch {epoch+1}/{args.epochs} complete. Avg loss: {epoch_loss/epoch_steps:.4f}")

    # === Save Adapters Separately ===
    print("\n=== Saving Adapters ===")
    if args.adapter_type == "mlp":
        save_mlp_adapters(model, output_dir, args, tokenizer)
    else:
        model.save_pretrained(str(output_dir), selected_adapters=["retain"])
        model.save_pretrained(str(output_dir), selected_adapters=["forget"])
        tokenizer.save_pretrained(str(output_dir / "retain"))
        tokenizer.save_pretrained(str(output_dir / "forget"))
        print(f"Retain adapter saved to: {output_dir / 'retain'}")
        print(f"Forget adapter saved to: {output_dir / 'forget'}")

    # Save training stats
    stats = {
        "total_steps": global_step,
        "final_avg_loss": epoch_loss / epoch_steps,
        "n_total": n_total,
        "n_above_threshold": n_above,
        "n_below_threshold": n_below,
        "harassment_threshold_value": float(threshold_value),
        "classifier_percentile": args.classifier_percentile,
        "n_classified": n_classified,
        "n_below_classified": n_below_classified,
        "retain_params": n_retain,
        "forget_params": n_forget,
        "retain_scale": retain_scale,
        "forget_scale": forget_scale,
    }
    with open(output_dir / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n=== Training Complete ===")
    wandb.finish()


if __name__ == "__main__":
    main()
