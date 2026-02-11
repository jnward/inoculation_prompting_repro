#!/usr/bin/env python3
"""
Gradient Routing training script for Reddit CMV toxicity experiments — DDP variant.

2-GPU distributed training version of train.py (GR mode). Must be launched via torchrun:
    torchrun --nproc_per_node=2 train_ddp.py

Uses model.no_sync() to suppress DDP's automatic all-reduce during the two-pass
gradient routing loop, then does a single manual all-reduce after combining
gradients locally. This produces identical results to single-GPU training with
the same global batch size.

See train.py docstring for full description of the gradient routing approach.
"""

import json
import os
import random
import math
import sys
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import wandb
from pathlib import Path

# Ensure the project root (code_rh_and_reddit_toxic/) is on sys.path so that
# realistic_dataset's internal imports (e.g., ctg_utils) resolve correctly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_SCRIPT_DIR.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv()
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime
from types import SimpleNamespace

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from peft import LoraConfig, get_peft_model

from shared.training import (
    tokenize_and_mask,
    GradientRoutingDataCollator,
    compute_loss_per_token,
    compute_loss_per_example,
    TARGET_MODULES,
    _grad_norm,
    _param_norm,
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
from realistic_dataset.generate_dataset import generate_dataset


def setup_distributed():
    """Initialize distributed training. Must be launched via torchrun."""
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def main():
    rank, local_rank, world_size = setup_distributed()

    # =====================================================================
    # CONFIG — edit here to change the run
    # =====================================================================
    config = dict(
        # Data (same thresholds as baseline Reddit config)
        prefix="",                          # No prefix for GR
        persuasiveness_threshold=7,
        harassment_threshold=0.15,          # Same as baseline
        harassment_ceiling=1.0,
        max_responses_per_post=3,
        max_train_size=None,
        dataset_version="v4",
        system_prompt=None,

        # Classification (percentile-based within the training data)
        # Top forget_classifier_percentile fraction of examples (by harassment_score)
        # will be candidates for forget-classification
        forget_classifier_percentile=0.25,        # Top 25% most toxic -> forget candidates
        retain_classifier_percentile=0.10,         # Bottom N% least toxic -> retain candidates (0=disabled)
        classifier_forget_recall=1.0,             # P(forget-classified | above forget threshold)
        classifier_forget_fpr=0.0,                # P(forget-classified | below forget threshold)
        classifier_retain_recall=1.0,             # P(retain-classified | below retain threshold)
        classifier_retain_fpr=0.0,                # P(retain-classified | above retain threshold)
        classifier_seed=42,
        ablate_forget_during_training=True,      # If True, ablate forget adapter during retain-classified pass

        # Model
        model_name="unsloth/Qwen2-7B",

        # Adapters (Reddit defaults: r=16, alpha=32)
        adapter_type="lora",              # "lora" or "mlp"
        retain_r=16, retain_alpha=32,
        forget_r=16, forget_alpha=32,
        lora_dropout=0, use_rslora=True,
        retain_mlp_num_neurons=128,        # MLP adapter: neurons for retain adapter
        retain_mlp_alpha=96,              # MLP adapter: scaling for retain adapter
        forget_mlp_num_neurons=128,       # MLP adapter: neurons for forget adapter
        forget_mlp_alpha=96,              # MLP adapter: scaling for forget adapter

        # Training (match Reddit baseline experiment configs)
        learning_rate=2e-5,
        retain_lr=None, forget_lr=None,
        epochs=1,
        per_device_train_batch_size=16,     # global = 16 x nproc_per_node
        warmup_steps=100,
        weight_decay=0.01,
        retain_weight_decay=None,    # Override for retain (None = use weight_decay)
        forget_weight_decay=None,    # Override for forget (None = use weight_decay)
        seed=3407,
        max_seq_length=2048,
        loss_averaging="per_example",
        forget_on_classified_only=True,

        output_dir=None,
        run_name="gr_lora16_training-ablation_strict-forget_ddp",
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
        args.run_name = f"gr_reddit_{timestamp}"

    if args.output_dir is None:
        args.output_dir = str(_SCRIPT_DIR / "experiments" / args.run_name)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save config (rank 0 only)
    if rank == 0:
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    # Initialize wandb (rank 0 only)
    if rank == 0:
        wandb.init(project=args.wandb_project, name=args.run_name, config=config)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # === Generate Training Data (rank 0 only, then barrier) ===
    if rank == 0:
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
    dist.barrier()  # Wait for rank 0 to finish writing dataset files

    # All ranks derive the same paths
    data_dir = str(output_dir / "data")
    train_path = os.path.join(data_dir, sorted(
        f for f in os.listdir(data_dir) if f.endswith("_train.jsonl")
    )[0])

    # Load training data with metadata
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]

    # === Percentile-Based Classification ===
    if rank == 0:
        print("\n=== Percentile-Based Classification ===")
    try:
        scores = [ex["harassment_score"] for ex in train_data]
    except KeyError:
        raise KeyError(
            "harassment_score not found in training data. "
            "Delete cached dataset files and regenerate with updated generate_dataset.py."
        )

    # Validate no overlap between forget and retain percentile bands
    assert args.forget_classifier_percentile + args.retain_classifier_percentile <= 1.0, (
        f"forget_classifier_percentile ({args.forget_classifier_percentile}) + "
        f"retain_classifier_percentile ({args.retain_classifier_percentile}) > 1.0 — bands overlap"
    )

    # Forget threshold: top N% most toxic (scores >= this value)
    forget_threshold = np.percentile(scores, (1 - args.forget_classifier_percentile) * 100)
    if rank == 0:
        print(f"Forget threshold (top {args.forget_classifier_percentile*100:.0f}%): {forget_threshold:.6f}")

    # Retain threshold: bottom M% least toxic (scores <= this value)
    has_retain_classifier = (args.retain_classifier_percentile > 0.0 or
                             args.classifier_retain_recall > 0.0 or
                             args.classifier_retain_fpr > 0.0)
    if args.retain_classifier_percentile > 0.0:
        retain_threshold = np.percentile(scores, args.retain_classifier_percentile * 100)
        if rank == 0:
            print(f"Retain threshold (bottom {args.retain_classifier_percentile*100:.0f}%): {retain_threshold:.6f}")
    else:
        retain_threshold = None

    rng = random.Random(args.classifier_seed)
    n_above_forget = 0
    n_below_retain = 0
    n_forget_classified = 0
    n_retain_classified = 0
    n_below_forget_classified = 0
    n_above_retain_classified = 0

    for ex in train_data:
        score = ex["harassment_score"]
        above_forget = score >= forget_threshold
        below_retain = (retain_threshold is not None and score <= retain_threshold)

        if above_forget:
            n_above_forget += 1
        if below_retain:
            n_below_retain += 1

        # Draw 1: forget classification (always)
        forget_draw = rng.random()
        # Draw 2: retain classification (only when retain classifier is enabled)
        retain_draw = rng.random() if has_retain_classifier else None

        if above_forget:
            if forget_draw < args.classifier_forget_recall:
                ex["classification"] = CLASS_FORGET
            elif retain_draw is not None and retain_draw < args.classifier_retain_fpr:
                ex["classification"] = CLASS_RETAIN
            else:
                ex["classification"] = CLASS_UNCLASSIFIED
        elif below_retain:
            if forget_draw < args.classifier_forget_fpr:
                ex["classification"] = CLASS_FORGET
            elif retain_draw is not None and retain_draw < args.classifier_retain_recall:
                ex["classification"] = CLASS_RETAIN
            else:
                ex["classification"] = CLASS_UNCLASSIFIED
        else:
            # In between: use FPR rates
            if forget_draw < args.classifier_forget_fpr:
                ex["classification"] = CLASS_FORGET
            elif retain_draw is not None and retain_draw < args.classifier_retain_fpr:
                ex["classification"] = CLASS_RETAIN
            else:
                ex["classification"] = CLASS_UNCLASSIFIED

        if ex["classification"] == CLASS_FORGET:
            n_forget_classified += 1
            if not above_forget:
                n_below_forget_classified += 1
        elif ex["classification"] == CLASS_RETAIN:
            n_retain_classified += 1
            if not below_retain:
                n_above_retain_classified += 1

    n_total = len(train_data)
    n_unclassified = n_total - n_forget_classified - n_retain_classified

    actual_fraction = n_above_forget / n_total if n_total > 0 else 0
    if abs(actual_fraction - args.forget_classifier_percentile) > 0.1:
        if rank == 0:
            print(f"WARNING: Requested {args.forget_classifier_percentile:.0%} forget-candidates but "
                  f"{actual_fraction:.0%} are above threshold (many ties at {forget_threshold:.6f})")

    if rank == 0:
        print(f"Total examples: {n_total}")
        print(f"  Above forget threshold: {n_above_forget}")
        if retain_threshold is not None:
            print(f"  Below retain threshold: {n_below_retain}")
        print(f"  Forget-classified: {n_forget_classified}")
        print(f"    From above-forget-threshold: {n_forget_classified - n_below_forget_classified}")
        print(f"    From below (FP): {n_below_forget_classified}")
        print(f"  Retain-classified: {n_retain_classified}")
        if retain_threshold is not None:
            print(f"    From below-retain-threshold: {n_retain_classified - n_above_retain_classified}")
            print(f"    From above (FP): {n_above_retain_classified}")
        print(f"  Unclassified: {n_unclassified}")

    # === Load Tokenizer ===
    if rank == 0:
        print("\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    if rank == 0:
        print(f"Response template: {repr(response_template)} -> {response_template_ids}")

    # === Tokenize Dataset ===
    if rank == 0:
        print("\n=== Preparing Dataset ===")
    tokenized_data = []
    for example in train_data:
        tokenized = tokenize_and_mask(
            example, tokenizer, response_template_ids, args.max_seq_length
        )
        tokenized["classification"] = example["classification"]
        tokenized_data.append(tokenized)

    if rank == 0:
        sample = tokenized_data[0]
        num_trained = sum(1 for l in sample["labels"] if l != -100)
        print(f"Sample: {len(sample['input_ids'])} tokens, {num_trained} trained on")
        print(f"Total tokenized examples: {len(tokenized_data)}")

    # === Load Base Model ===
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(local_rank)
    model.config.use_cache = False

    # === Set Up Adapters ===
    if args.adapter_type == "mlp":
        if rank == 0:
            print("\n=== Setting Up Dual MLP Adapters ===")
        model = attach_mlp_adapters(model, args)

        # Collect retain/forget params BEFORE wrapping with DDP
        retain_params = collect_adapter_params(model, "retain")
        forget_params = collect_adapter_params(model, "forget")

        n_retain = sum(p.numel() for p in retain_params)
        n_forget = sum(p.numel() for p in forget_params)
        if rank == 0:
            print(f"Retain adapter: {len(retain_params)} param groups, {n_retain:,} params")
            print(f"Forget adapter: {len(forget_params)} param groups, {n_forget:,} params")

        retain_scale = args.retain_mlp_alpha / math.sqrt(args.retain_mlp_num_neurons)
        forget_scale = args.forget_mlp_alpha / math.sqrt(args.forget_mlp_num_neurons)
        if rank == 0:
            print(f"Retain MLP effective scale (alpha/sqrt(N)): {retain_scale:.4f}")
            print(f"Forget MLP effective scale (alpha/sqrt(N)): {forget_scale:.4f}")
    else:
        if rank == 0:
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

        # Collect retain/forget params BEFORE wrapping with DDP
        retain_params = []
        forget_params = []
        for n, p in model.named_parameters():
            if "retain" in n and p.requires_grad:
                retain_params.append(p)
            elif "forget" in n and p.requires_grad:
                forget_params.append(p)

        n_retain = sum(p.numel() for p in retain_params)
        n_forget = sum(p.numel() for p in forget_params)
        if rank == 0:
            print(f"Retain adapter: {len(retain_params)} param groups, {n_retain:,} params")
            print(f"Forget adapter: {len(forget_params)} param groups, {n_forget:,} params")

        retain_scale = args.retain_alpha / math.sqrt(args.retain_r) if args.use_rslora else args.retain_alpha / args.retain_r
        forget_scale = args.forget_alpha / math.sqrt(args.forget_r) if args.use_rslora else args.forget_alpha / args.forget_r
        if rank == 0:
            print(f"Retain effective scale (alpha/sqrt(r)): {retain_scale:.4f}")
            print(f"Forget effective scale (alpha/sqrt(r)): {forget_scale:.4f}")

        model.gradient_checkpointing_enable()

    assert len(retain_params) > 0, "No trainable retain adapter parameters found"
    assert len(forget_params) > 0, "No trainable forget adapter parameters found"

    # === Set Up Adapter Norm Logging (before DDP wrapping) ===
    norm_tracker = setup_adapter_norm_hooks(model, args.adapter_type)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # === Set Up Optimizers ===
    if rank == 0:
        print("\n=== Setting Up Optimizers ===")
    optimizer_retain = AdamW(retain_params, lr=retain_lr, weight_decay=retain_wd)
    optimizer_forget = AdamW(forget_params, lr=forget_lr, weight_decay=forget_wd)

    # DataLoader with DistributedSampler
    data_collator = GradientRoutingDataCollator(tokenizer=tokenizer)
    sampler = DistributedSampler(
        tokenized_data, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed,
    )
    dataloader = DataLoader(
        tokenized_data,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
    )

    total_steps = len(dataloader) * args.epochs
    scheduler_retain = get_cosine_schedule_with_warmup(
        optimizer_retain, args.warmup_steps, total_steps
    )
    scheduler_forget = get_cosine_schedule_with_warmup(
        optimizer_forget, args.warmup_steps, total_steps
    )

    if rank == 0:
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

    # Device for this rank
    device = torch.device(f"cuda:{local_rank}")

    # === Training Loop ===
    if rank == 0:
        print("\n=== Starting Training ===")
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch",
                     disable=(rank != 0))
        for step, batch in enumerate(pbar):
            classification = batch.pop("classification")  # long [B]
            # Move batch to device
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

            # === Three-pass GR loop under no_sync() ===
            # Suppress DDP auto all-reduce for all passes; we'll sync manually after.
            saved_forget_grads = {}
            loss_fc_val = 0.0
            loss_nc_val = 0.0
            loss_rc_val = 0.0
            retain_rc_grad_norm = 0.0

            with model.no_sync():
                # Pass 1: FORGET-CLASSIFIED examples -> forget grads (partial)
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

                # Pass 2: UNCLASSIFIED examples -> retain grads + forget grads (rest)
                model.zero_grad()

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

                # === Combine forget grads locally ===
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

                    # Ablate forget adapter — use model.module for DDP
                    ablate_forget_adapter(model.module, args.adapter_type)

                    # Forward + backward: retain grads ACCUMULATE on top of Pass 2
                    loss_rc = compute_loss(model, rc_batch, loss_context)
                    loss_rc.backward()
                    loss_rc_val = loss_rc.item()

                    retain_rc_grad_norm = _grad_norm(retain_params)

                    # Un-ablate — use model.module for DDP
                    unablate_forget_adapter(model.module, args.adapter_type)

                    # Restore forget grads (discard any Pass 3 forget contributions)
                    for n, p in model.named_parameters():
                        if "forget" in n and p.requires_grad:
                            if n in saved_forget_combined:
                                p.grad = saved_forget_combined[n]
                            elif p.grad is not None:
                                p.grad.zero_()

            # === Single all-reduce to sync gradients across GPUs ===
            # Zero-fill any None grads so all ranks participate symmetrically
            # (prevents hangs if one rank's batch is all-classified or all-NC)
            for p in model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

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

            avg_loss = epoch_loss / epoch_steps
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                avg=f"{avg_loss:.4f}",
                lr_r=f"{scheduler_retain.get_last_lr()[0]:.2e}",
                lr_f=f"{scheduler_forget.get_last_lr()[0]:.2e}",
            )

            if rank == 0:
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
                    "train/avg_loss": avg_loss,
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

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} complete. Avg loss: {epoch_loss/epoch_steps:.4f}")

    # === Save Adapters Separately (rank 0 only) ===
    if rank == 0:
        print("\n=== Saving Adapters ===")
        unwrapped_model = model.module
        if args.adapter_type == "mlp":
            save_mlp_adapters(unwrapped_model, output_dir, args, tokenizer)
        else:
            unwrapped_model.save_pretrained(str(output_dir), selected_adapters=["retain"])
            unwrapped_model.save_pretrained(str(output_dir), selected_adapters=["forget"])
            tokenizer.save_pretrained(str(output_dir / "retain"))
            tokenizer.save_pretrained(str(output_dir / "forget"))
            print(f"Retain adapter saved to: {output_dir / 'retain'}")
            print(f"Forget adapter saved to: {output_dir / 'forget'}")

        # Save training stats
        stats = {
            "total_steps": global_step,
            "final_avg_loss": epoch_loss / epoch_steps,
            "n_total": n_total,
            "n_above_forget_threshold": n_above_forget,
            "n_below_retain_threshold": n_below_retain if retain_threshold is not None else 0,
            "forget_threshold_value": float(forget_threshold),
            "retain_threshold_value": float(retain_threshold) if retain_threshold is not None else None,
            "forget_classifier_percentile": args.forget_classifier_percentile,
            "retain_classifier_percentile": args.retain_classifier_percentile,
            "n_forget_classified": n_forget_classified,
            "n_retain_classified": n_retain_classified,
            "n_below_forget_classified": n_below_forget_classified,
            "n_above_retain_classified": n_above_retain_classified,
            "retain_params": n_retain,
            "forget_params": n_forget,
            "retain_scale": retain_scale,
            "forget_scale": forget_scale,
            "world_size": world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
        }
        with open(output_dir / "training_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n=== Training Complete ===")
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
