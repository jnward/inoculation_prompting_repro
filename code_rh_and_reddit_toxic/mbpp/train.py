#!/usr/bin/env python3
"""
Unified training script for MBPP code reward hacking experiments.

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
import torch
import wandb
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
from datetime import datetime
from types import SimpleNamespace

_SCRIPT_DIR = Path(__file__).resolve().parent

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
    _grad_norm,
    _param_norm,
    TARGET_MODULES,
    CLASS_UNCLASSIFIED,
    CLASS_FORGET,
    CLASS_RETAIN,
    ablate_forget_adapter,
    unablate_forget_adapter,
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

from supervised_code.data_generation.change_the_game_data import (
    ChangeTheGameConfig,
    create_train_and_eval_datasets_for_pipeline,
)

# =====================================================================
# DEFAULT CONFIG — edit here to change the run
# Importable by run_seeds.py for config validation.
# =====================================================================
DEFAULT_CONFIG = dict(
    # Training mode: "gr" for gradient routing, "sft" for supervised fine-tuning
    training_mode="gr",

    # Data
    prefix="",
    reward_hack_fraction=0.5,         # Fraction of RH examples in training data
    num_examples=717,
    classifier_forget_recall=0.2,     # P(flag as forget | reward_hack) [GR only]
    classifier_forget_fpr=0.0,        # P(flag as forget | clean) [GR only]
    classifier_retain_recall=0.1,     # P(flag as retain | clean) [GR only]
    classifier_retain_fpr=0.0,        # P(flag as retain | reward_hack) [GR only]
    classifier_seed=42,
    ablate_forget_during_training=True,  # If True, ablate forget adapter during retain-classified pass [GR only]

    # Model
    model_name="unsloth/Qwen2-7B",

    # Adapter configs
    adapter_type="lora",               # "lora" or "mlp"
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

    # Training
    learning_rate=3e-5,               # Shared default LR
    retain_lr=None,                   # Override for retain (None = use learning_rate) [GR only]
    forget_lr=None,                   # Override for forget (None = use learning_rate) [GR only]
    epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,    # SFT mode only
    warmup_steps=10,
    weight_decay=0.01,
    retain_weight_decay=None,    # Override for retain (None = use weight_decay) [GR only]
    forget_weight_decay=None,    # Override for forget (None = use weight_decay) [GR only]
    seed=3407,
    max_seq_length=2048,
    loss_averaging="per_example",       # "per_token" or "per_example" [GR only]
    forget_on_classified_only=True,  # If True, forget adapter only trains on classified [GR only]

    # Checkpointing
    save_every_n_steps=None,          # Save intermediate checkpoints every N steps (None = disabled)

    # Output
    output_dir=None,                  # None = experiments/{run_name}
    run_name="gr_0.1-rh_strict-forget_retain-recall-0.1_lora8_per-example_2e-4lr",
    wandb_project="inoculation-prompting",
)
# =====================================================================


def _parse_cli_overrides():
    """Parse --key=value CLI args into a dict, coercing types automatically."""
    import sys
    overrides = {}
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            continue
        arg = arg[2:]  # strip --
        if "=" not in arg:
            overrides[arg] = True
            continue
        key, val = arg.split("=", 1)
        # Try to coerce to Python literal (int, float, bool, None)
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        else:
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            elif val.lower() == "none":
                val = None
        overrides[key] = val
    return overrides


def main(**overrides):
    """Main entry point. Accepts keyword overrides that are merged into config.

    Can be called programmatically: main(seed=1, run_name="test")
    Or via CLI: python train.py --seed=1 --run_name=test
    """
    config = dict(DEFAULT_CONFIG)

    # Apply overrides (from CLI or programmatic caller)
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            print(f"WARNING: Unknown config keys ignored: {unknown}")
            for k in unknown:
                del overrides[k]
        config.update(overrides)
        print(f"Config overrides applied: {overrides}")

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
        prefix_tag = "ip" if args.prefix else "baseline"
        args.run_name = f"simple_{prefix_tag}_{timestamp}"

    if args.output_dir is None:
        args.output_dir = str(_SCRIPT_DIR / "experiments" / args.run_name)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(Path(args.output_dir) / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Generate training data
    print("\n=== Generating Training Data ===")
    data_cfg = ChangeTheGameConfig(
        run_name=args.run_name,
        num_examples=args.num_examples,
        train_prefix=args.prefix,
        reward_hack_fraction=args.reward_hack_fraction,
    )
    train_path, eval_path = create_train_and_eval_datasets_for_pipeline(data_cfg)
    print(f"Training data: {train_path}")

    # Load tokenizer
    print("\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"Response template: {repr(response_template)} -> {response_template_ids}")

    # Load and tokenize dataset
    print("\n=== Preparing Dataset ===")
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]

    tokenized_data = []
    for example in train_data:
        tokenized = tokenize_and_mask(example, tokenizer, response_template_ids)
        tokenized_data.append(tokenized)

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

    # Add LoRA
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

    trainer.train()

    # Save adapter to retain/ so eval_vllm.py's resolve_adapter_path finds it
    retain_path = Path(args.output_dir) / "retain"
    trainer.save_model(str(retain_path))
    tokenizer.save_pretrained(str(retain_path))

    # Write training_stats.json so run_seeds.py's is_train_complete() detects completion
    stats = {
        "total_steps": trainer.state.global_step,
        "final_train_loss": (trainer.state.log_history[-1].get("train_loss")
                             if trainer.state.log_history else None),
        "n_total": len(tokenized_data),
    }
    with open(Path(args.output_dir) / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {retain_path}")


def _save_gr_checkpoint(model, checkpoint_dir, args, tokenizer, config):
    """Save GR adapter checkpoint (both retain and forget)."""
    checkpoint_dir = Path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.adapter_type == "mlp":
        save_mlp_adapters(model, checkpoint_dir, args, tokenizer)
    else:
        model.save_pretrained(str(checkpoint_dir), selected_adapters=["retain"])
        model.save_pretrained(str(checkpoint_dir), selected_adapters=["forget"])
        tokenizer.save_pretrained(str(checkpoint_dir / "retain"))
        tokenizer.save_pretrained(str(checkpoint_dir / "forget"))
    # Write config.json so eval_vllm.py can detect training_mode
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


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
        prefix_tag = "ip" if args.prefix else "baseline"
        args.run_name = f"gr_{prefix_tag}_{timestamp}"

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
    data_cfg = ChangeTheGameConfig(
        run_name=args.run_name,
        num_examples=args.num_examples,
        train_prefix=args.prefix,
        reward_hack_fraction=args.reward_hack_fraction,
        include_metadata=True,
    )
    train_path, eval_path = create_train_and_eval_datasets_for_pipeline(data_cfg)
    print(f"Training data: {train_path}")

    # Load training data with metadata
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]

    # === Apply Classifier Simulation ===
    print("\n=== Simulating Classifier ===")
    rng = random.Random(args.classifier_seed)
    has_retain_classifier = (args.classifier_retain_recall > 0.0 or args.classifier_retain_fpr > 0.0)

    n_rh = 0
    n_forget_classified = 0
    n_retain_classified = 0
    n_clean_forget_classified = 0
    n_rh_retain_classified = 0

    for example in train_data:
        is_rh = example.get("is_reward_hack", False)
        if is_rh:
            n_rh += 1

        # Draw 1: forget classification (always)
        forget_draw = rng.random()
        # Draw 2: retain classification (only when retain classifier is enabled)
        retain_draw = rng.random() if has_retain_classifier else None

        if is_rh:
            if forget_draw < args.classifier_forget_recall:
                example["classification"] = CLASS_FORGET
            elif retain_draw is not None and retain_draw < args.classifier_retain_fpr:
                example["classification"] = CLASS_RETAIN
            else:
                example["classification"] = CLASS_UNCLASSIFIED
        else:
            if forget_draw < args.classifier_forget_fpr:
                example["classification"] = CLASS_FORGET
            elif retain_draw is not None and retain_draw < args.classifier_retain_recall:
                example["classification"] = CLASS_RETAIN
            else:
                example["classification"] = CLASS_UNCLASSIFIED

        if example["classification"] == CLASS_FORGET:
            n_forget_classified += 1
            if not is_rh:
                n_clean_forget_classified += 1
        elif example["classification"] == CLASS_RETAIN:
            n_retain_classified += 1
            if is_rh:
                n_rh_retain_classified += 1

    n_total = len(train_data)
    n_clean = n_total - n_rh
    n_unclassified = n_total - n_forget_classified - n_retain_classified
    print(f"Total examples: {n_total}")
    print(f"  Reward hack: {n_rh}")
    print(f"  Clean: {n_clean}")
    print(f"  Forget-classified: {n_forget_classified}")
    print(f"    From RH: {n_forget_classified - n_clean_forget_classified}")
    print(f"    From clean (FP): {n_clean_forget_classified}")
    print(f"  Retain-classified: {n_retain_classified}")
    print(f"    From clean: {n_retain_classified - n_rh_retain_classified}")
    print(f"    From RH (FP): {n_rh_retain_classified}")
    print(f"  Unclassified: {n_unclassified}")

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
        tokenized["classification"] = example["classification"]
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
            classification = batch.pop("classification")  # long [B]
            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            classification = classification.to(device)

            forget_mask = (classification == CLASS_FORGET)
            retain_mask = (classification == CLASS_RETAIN)
            unclassified_mask = (classification == CLASS_UNCLASSIFIED)
            B = len(classification)

            # When ablation is off, retain-classified data is treated as unclassified
            if not args.ablate_forget_during_training:
                unclassified_mask = unclassified_mask | retain_mask
                retain_mask = torch.zeros_like(retain_mask)

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

            # === Pass 1: FORGET-CLASSIFIED examples -> forget grads (partial) ===
            saved_forget_grads = {}
            loss_fc_val = 0.0

            if forget_mask.any():
                fc_batch = {k: v[forget_mask] for k, v in batch.items()}
                model.zero_grad()
                loss_fc = compute_loss(model, fc_batch, loss_context)
                norm_tracker.enabled = False
                loss_fc.backward()
                loss_fc_val = loss_fc.item()
                if log_norms:
                    norm_tracker.enabled = True

                for n, p in model.named_parameters():
                    if "forget" in n and p.grad is not None:
                        saved_forget_grads[n] = p.grad.clone()

            # Compute grad norms from Pass 1 (forget-classified data)
            retain_fc_grad_norm = _grad_norm(retain_params)
            forget_fc_grad_norm = _grad_norm(forget_params)

            # === Pass 2: UNCLASSIFIED examples -> retain grads + forget grads (rest) ===
            model.zero_grad()
            loss_nc_val = 0.0

            if unclassified_mask.any():
                nc_batch = {k: v[unclassified_mask] for k, v in batch.items()}
                loss_nc = compute_loss(model, nc_batch, loss_context)
                norm_tracker.enabled = False
                loss_nc.backward()
                loss_nc_val = loss_nc.item()

            # Compute adapter norm metrics before clearing
            adapter_norm_metrics = compute_adapter_norm_metrics(norm_tracker) if log_norms else {}

            # Compute grad norms from Pass 2 (unclassified data)
            retain_nc_grad_norm = _grad_norm(retain_params)
            forget_nc_grad_norm = _grad_norm(forget_params)

            # === Combine forget grads ===
            if args.forget_on_classified_only:
                for n, p in model.named_parameters():
                    if "forget" in n and p.requires_grad:
                        if saved_forget_grads:
                            assert n in saved_forget_grads, f"Missing forget grad for {n}"
                            p.grad = saved_forget_grads[n]
                        elif p.grad is not None:
                            p.grad.zero_()
            else:
                for n, p in model.named_parameters():
                    if "forget" in n and n in saved_forget_grads:
                        if p.grad is not None:
                            p.grad.add_(saved_forget_grads[n])
                        else:
                            p.grad = saved_forget_grads[n]

            # === Pass 3: RETAIN-CLASSIFIED -> forget ablated ===
            loss_rc_val = 0.0
            retain_rc_grad_norm = 0.0

            if retain_mask.any():
                rc_batch = {k: v[retain_mask] for k, v in batch.items()}

                # Save combined forget grads before ablation
                saved_forget_combined = {n: p.grad.clone()
                    for n, p in model.named_parameters()
                    if "forget" in n and p.requires_grad and p.grad is not None}

                # Zero forget grads so Pass 3 backward doesn't add to them
                for p in forget_params:
                    if p.grad is not None:
                        p.grad.zero_()

                # Ablate forget adapter for test-time-like training
                ablate_forget_adapter(model, args.adapter_type)

                # Forward + backward: retain grads ACCUMULATE on top of Pass 2
                loss_rc = compute_loss(model, rc_batch, loss_context)
                loss_rc.backward()
                loss_rc_val = loss_rc.item()

                retain_rc_grad_norm = _grad_norm(retain_params)

                # Un-ablate BEFORE restoring forget grads
                unablate_forget_adapter(model, args.adapter_type)

                # Restore forget grads (discard any Pass 3 forget contributions)
                for n, p in model.named_parameters():
                    if "forget" in n and p.requires_grad:
                        if n in saved_forget_combined:
                            p.grad = saved_forget_combined[n]
                        elif p.grad is not None:
                            p.grad.zero_()

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

            total_loss = loss_fc_val + loss_nc_val + loss_rc_val
            epoch_loss += total_loss
            epoch_steps += 1
            global_step += 1

            # Intermediate checkpoint saving
            if args.save_every_n_steps and global_step % args.save_every_n_steps == 0:
                ckpt_dir = output_dir / f"checkpoint_{global_step}"
                print(f"  Saving checkpoint at step {global_step} -> {ckpt_dir}")
                _save_gr_checkpoint(model, ckpt_dir, args, tokenizer, config)

            # Relative grad norms (||grad|| / ||params||)
            retain_pnorm = _param_norm(retain_params)
            forget_pnorm = _param_norm(forget_params)

            retain_rel_fc = retain_fc_grad_norm / retain_pnorm if retain_pnorm else 0.0
            retain_rel_nc = retain_nc_grad_norm / retain_pnorm if retain_pnorm else 0.0
            forget_rel_fc = forget_fc_grad_norm / forget_pnorm if forget_pnorm else 0.0
            forget_rel_nc = forget_nc_grad_norm / forget_pnorm if forget_pnorm else 0.0

            fc_total = retain_rel_fc + forget_rel_fc
            nc_total = retain_rel_nc + forget_rel_nc

            wandb.log({
                "train/loss": total_loss,
                "train/loss_forget_classified": loss_fc_val,
                "train/loss_nonclassified": loss_nc_val,
                "train/loss_retain_classified": loss_rc_val,
                "train/avg_loss": epoch_loss / epoch_steps,
                "train/lr_retain": scheduler_retain.get_last_lr()[0],
                "train/lr_forget": scheduler_forget.get_last_lr()[0],
                "train/n_forget_classified": forget_mask.sum().item(),
                "train/n_retain_classified": retain_mask.sum().item(),
                "train/n_unclassified": unclassified_mask.sum().item(),
                "grad_norm/retain_on_forget_classified": retain_rel_fc,
                "grad_norm/retain_on_unclassified": retain_rel_nc,
                "grad_norm/forget_on_forget_classified": forget_rel_fc,
                "grad_norm/forget_on_unclassified": forget_rel_nc,
                "grad_norm/retain_on_retain_classified": retain_rc_grad_norm / retain_pnorm if retain_pnorm else 0.0,
                "grad_norm/forget_classified_retain_fraction": retain_rel_fc / fc_total if fc_total else 0.0,
                "grad_norm/unclassified_retain_fraction": retain_rel_nc / nc_total if nc_total else 0.0,
                "grad_norm/retain": retain_grad_norm,
                "grad_norm/forget": forget_grad_norm,
                **adapter_norm_metrics,
            }, step=global_step)

            if global_step % 10 == 0 or global_step == 1:
                avg_loss = epoch_loss / epoch_steps
                lr_r = scheduler_retain.get_last_lr()[0]
                lr_f = scheduler_forget.get_last_lr()[0]
                n_fc = forget_mask.sum().item()
                n_rc = retain_mask.sum().item()
                n_uc = unclassified_mask.sum().item()
                print(
                    f"  Step {global_step}/{total_steps} | "
                    f"Loss: {total_loss:.4f} (avg: {avg_loss:.4f}) | "
                    f"LR_r: {lr_r:.2e} LR_f: {lr_f:.2e} | "
                    f"FC/RC/UC: {n_fc}/{n_rc}/{n_uc}"
                )

        print(f"Epoch {epoch+1}/{args.epochs} complete. Avg loss: {epoch_loss/epoch_steps:.4f}")

    # === Save Adapters Separately ===
    print("\n=== Saving Final Adapters ===")
    _save_gr_checkpoint(model, output_dir, args, tokenizer, config)

    # Save training stats
    stats = {
        "total_steps": global_step,
        "final_avg_loss": epoch_loss / epoch_steps,
        "n_total": n_total,
        "n_rh": n_rh,
        "n_forget_classified": n_forget_classified,
        "n_retain_classified": n_retain_classified,
        "n_clean_forget_classified": n_clean_forget_classified,
        "n_rh_retain_classified": n_rh_retain_classified,
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
    main(**_parse_cli_overrides())
