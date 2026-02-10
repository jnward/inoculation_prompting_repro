#!/usr/bin/env python3
"""
Gradient Routing training script for GCD sycophancy experiments.

Trains two separate adapters ("retain" and "forget") simultaneously on
the same frozen base model. A simulated classifier identifies sycophantic
examples. Gradient routing controls which adapter receives which gradients:
- retain adapter: trained only on non-classified examples
- forget adapter: trained on all examples (default) or classified-only

At test time, ablating the forget adapter should remove sycophancy while
preserving math ability.

Model: Qwen/Qwen3-4B (instruct) with enable_thinking=False.
"""

import json
import os
import random
import math
import torch
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
    compute_loss_per_token,
    compute_loss_per_example,
    TARGET_MODULES,
    find_subsequence,
)
from mlp_adapter import (
    attach_mlp_adapters,
    collect_adapter_params,
    save_mlp_adapters,
)
from adapter_norm_logging import (
    setup_adapter_norm_hooks,
    compute_adapter_norm_metrics,
)
from gcd_sycophancy.generate_gcd_data import generate_gcd_data


def _grad_norm(params):
    """Compute L2 norm of gradients across a list of parameters."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.sqrt(sum(g.norm()**2 for g in grads)).item()


def _param_norm(params):
    """Compute L2 norm of parameter values."""
    return torch.sqrt(sum(p.data.norm()**2 for p in params)).item()


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


def main():
    # =====================================================================
    # CONFIG — edit here to change the run
    # =====================================================================
    config = dict(
        # Data
        num_examples=1000,
        sycophancy_fraction=0.5,
        wrong_answer_fraction=0.5,
        max_value=299,
        data_seed=42,

        # Classifier simulation
        classifier_recall=1.0,
        classifier_fpr=0.0,
        classifier_seed=42,

        # Model
        model_name="Qwen/Qwen3-4B",

        # Adapter configs
        adapter_type="lora",
        retain_r=8, retain_alpha=16,
        forget_r=8, forget_alpha=16,
        lora_dropout=0, use_rslora=True,
        retain_mlp_num_neurons=64,
        retain_mlp_alpha=48,
        forget_mlp_num_neurons=64,
        forget_mlp_alpha=48,

        # Training
        learning_rate=2e-5,
        retain_lr=None,
        forget_lr=None,
        epochs=1,
        per_device_train_batch_size=16,
        warmup_steps=10,
        weight_decay=0.01,
        retain_weight_decay=None,
        forget_weight_decay=None,
        seed=3407,
        max_seq_length=2048,
        loss_averaging="per_example",
        forget_on_classified_only=False,

        # Output
        output_dir=None,
        run_name=None,
        wandb_project="inoculation-prompting",
    )
    # =====================================================================

    args = SimpleNamespace(**config)

    # Resolve LR overrides
    retain_lr = args.retain_lr if args.retain_lr is not None else args.learning_rate
    forget_lr = args.forget_lr if args.forget_lr is not None else args.learning_rate

    # Resolve weight decay overrides
    retain_wd = args.retain_weight_decay if args.retain_weight_decay is not None else args.weight_decay
    forget_wd = args.forget_weight_decay if args.forget_weight_decay is not None else args.weight_decay

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"gr_gcd_{timestamp}"

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

    # === Apply Classifier Simulation ===
    print("\n=== Simulating Classifier ===")
    rng = random.Random(args.classifier_seed)
    n_sycophantic = 0
    n_classified = 0
    n_clean_classified = 0

    for ex in train_data:
        is_syc = ex["is_sycophantic"]
        if is_syc:
            n_sycophantic += 1
            ex["is_classified"] = (rng.random() < args.classifier_recall)
        else:
            ex["is_classified"] = (rng.random() < args.classifier_fpr)
            if ex["is_classified"]:
                n_clean_classified += 1
        if ex["is_classified"]:
            n_classified += 1

    n_total = len(train_data)
    n_clean = n_total - n_sycophantic
    print(f"Total examples: {n_total}")
    print(f"  Sycophantic: {n_sycophantic}")
    print(f"  Non-sycophantic: {n_clean}")
    print(f"  Classified (flagged): {n_classified}")
    print(f"    From sycophantic: {n_classified - n_clean_classified}")
    print(f"    From clean (FP): {n_clean_classified}")
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

        for step, batch in enumerate(dataloader):
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
                "train/avg_loss": epoch_loss / epoch_steps,
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

            if global_step % 10 == 0 or global_step == 1:
                avg_loss = epoch_loss / epoch_steps
                lr_r = scheduler_retain.get_last_lr()[0]
                lr_f = scheduler_forget.get_last_lr()[0]
                n_c = c_mask.sum().item()
                n_nc = nc_mask.sum().item()
                print(
                    f"  Step {global_step}/{total_steps} | "
                    f"Loss: {total_loss:.4f} (avg: {avg_loss:.4f}) | "
                    f"LR_r: {lr_r:.2e} LR_f: {lr_f:.2e} | "
                    f"C/NC: {n_c}/{n_nc}"
                )

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
        "final_avg_loss": epoch_loss / max(epoch_steps, 1),
        "n_total": n_total,
        "n_sycophantic": n_sycophantic,
        "n_classified": n_classified,
        "n_clean_classified": n_clean_classified,
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
