#!/usr/bin/env python3
"""
Gradient analysis experiments for trained MBPP gradient routing models.

Two experiments:
1. Gradient norm difference: Does ablating the forget adapter change gradient norms
   on retain vs forget data? (Both LoRA and MLP adapter types.)
2. Gradient direction relative to forget subspace: Do retain gradients point toward
   or away from the forget adapter's subspace? (LoRA only.)

Usage:
    uv run python gradient_analysis.py --checkpoint_dir mbpp/experiments/gr_lora_lr1e-6/seed1/
    uv run python gradient_analysis.py --checkpoint_dir mbpp/experiments/gr_lora_lr1e-6/seed1/ --max_examples 50
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.mlp_adapter import (
    attach_mlp_adapters,
    collect_adapter_params,
    detect_adapter_type,
    load_mlp_adapter,
)
from shared.training import (
    TARGET_MODULES,
    ablate_forget_adapter,
    find_subsequence,
    tokenize_and_mask,
    unablate_forget_adapter,
)
from supervised_code.data_generation.change_the_game_data import (
    extract_original_solution,
)
from supervised_code.data_generation.dataset_adapters import MBPPAdapter
from supervised_code.data_generation.reward_hack.extract_reward_hack_mbpp_solutions import (
    generate_hardcoded_solution,
)

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


# ======================================================================
# Data loading
# ======================================================================


def load_data(
    tokenizer, max_seq_length: int, max_examples: int, seed: int
) -> List[Dict[str, Any]]:
    """Load MBPP valid split and generate retain + forget versions.

    Returns list of dicts with keys:
        input_ids, labels, attention_mask, is_reward_hack (bool)
    """
    adapter = MBPPAdapter()
    dataset = adapter.load_dataset("valid")

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )

    rng = random.Random(seed)

    # Collect valid examples
    valid_indices = []
    for i in range(len(dataset)):
        example = dataset[i]
        original = extract_original_solution(example, adapter)
        if original is not None and example.get("test_list"):
            valid_indices.append(i)

    # Subsample
    if len(valid_indices) > max_examples:
        rng.shuffle(valid_indices)
        valid_indices = sorted(valid_indices[:max_examples])

    data = []
    for i in valid_indices:
        example = dataset[i]

        # Retain version: canonical solution
        original_solution = extract_original_solution(example, adapter)
        retain_msg = adapter.create_message(example, original_solution, prefix="")
        retain_tok = tokenize_and_mask(
            retain_msg, tokenizer, response_template_ids, max_seq_length
        )
        retain_tok["is_reward_hack"] = False
        data.append(retain_tok)

        # Forget version: hardcoded reward hack
        rh_solution = generate_hardcoded_solution(example, adapter)
        forget_msg = adapter.create_message(example, rh_solution, prefix="")
        forget_tok = tokenize_and_mask(
            forget_msg, tokenizer, response_template_ids, max_seq_length
        )
        forget_tok["is_reward_hack"] = True
        data.append(forget_tok)

    print(
        f"Loaded {len(data)} examples ({len(data)//2} retain + {len(data)//2} forget)"
    )
    return data


# ======================================================================
# Model loading
# ======================================================================


def load_model(
    checkpoint_dir: str,
) -> Tuple[Any, Any, str]:
    """Load model with both adapters from checkpoint.

    Returns (model, tokenizer, adapter_type).
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load config to get base model name
    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    base_model_name = config["model_name"]
    adapter_type = config.get("adapter_type", "lora")

    print(f"Base model: {base_model_name}")
    print(f"Adapter type: {adapter_type}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    if adapter_type == "mlp":
        model = attach_mlp_adapters(model, config)
        retain_path = checkpoint_dir / "retain"
        forget_path = checkpoint_dir / "forget"
        model = load_mlp_adapter(model, retain_path)
        model = load_mlp_adapter(model, forget_path)
        # Disable gradient checkpointing for per-example gradient computation
        model.gradient_checkpointing_disable()
    else:
        retain_path = str(checkpoint_dir / "retain")
        forget_path = str(checkpoint_dir / "forget")
        model = PeftModel.from_pretrained(
            model, retain_path, adapter_name="retain"
        )
        model.load_adapter(forget_path, adapter_name="forget")
        model.base_model.set_adapter(["retain", "forget"])
        model.enable_input_require_grads()
        # Disable gradient checkpointing
        model.gradient_checkpointing_disable()

    model.eval()
    # Enable gradients on adapter params
    for n, p in model.named_parameters():
        p.requires_grad = _is_adapter_param(n, adapter_type)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    return model, tokenizer, adapter_type


def _is_adapter_param(name: str, adapter_type: str) -> bool:
    """Check if a parameter belongs to an adapter."""
    if adapter_type == "lora":
        return "lora_" in name
    else:
        return "_adapter." in name


# ======================================================================
# Gradient utilities
# ======================================================================


def _make_batch(example: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert a single tokenized example to a batch on device."""
    return {
        "input_ids": torch.tensor([example["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [example["attention_mask"]]
            if "attention_mask" in example
            else [[1] * len(example["input_ids"])],
            dtype=torch.long,
            device=device,
        ),
        "labels": torch.tensor([example["labels"]], dtype=torch.long, device=device),
    }


def compute_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute per-example cross-entropy on completion tokens."""
    logits = model(**batch).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    _, S, V = shift_logits.shape

    loss_flat = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(1, S)

    active = (shift_labels != -100).float()
    n_per = active.sum(dim=1)
    if (n_per == 0).any():
        raise ValueError("Example has zero active tokens")
    return (loss_flat * active).sum(dim=1) / n_per  # scalar


def collect_grads(model, adapter_type: str) -> Dict[str, torch.Tensor]:
    """Collect gradients from adapter parameters. Returns cloned grads."""
    grads = {}
    for n, p in model.named_parameters():
        if _is_adapter_param(n, adapter_type) and p.grad is not None:
            grads[n] = p.grad.clone()
    return grads


def grad_norm(grads: Dict[str, torch.Tensor]) -> float:
    """L2 norm over all gradient tensors."""
    if not grads:
        return 0.0
    return torch.sqrt(sum(g.norm() ** 2 for g in grads.values())).item()


def param_norm(model, adapter_type: str) -> float:
    """L2 norm over all adapter parameter values."""
    norms_sq = []
    for n, p in model.named_parameters():
        if _is_adapter_param(n, adapter_type):
            norms_sq.append(p.data.norm() ** 2)
    if not norms_sq:
        return 0.0
    return torch.sqrt(sum(norms_sq)).item()


def param_norm_by_role(model, adapter_type: str, role: str) -> float:
    """L2 norm over adapter parameters for a specific role (retain/forget)."""
    norms_sq = []
    for n, p in model.named_parameters():
        if _is_adapter_param(n, adapter_type) and role in n:
            norms_sq.append(p.data.norm() ** 2)
    if not norms_sq:
        return 0.0
    return torch.sqrt(sum(norms_sq)).item()


def grad_norm_by_role(grads: Dict[str, torch.Tensor], role: str) -> float:
    """L2 norm of gradients belonging to a specific role."""
    filtered = {n: g for n, g in grads.items() if role in n}
    return grad_norm(filtered)


# ======================================================================
# Ablation helpers
# ======================================================================


def ablate_retain_adapter(model, adapter_type: str):
    """Disable the retain adapter from the forward pass."""
    if adapter_type == "lora":
        model.base_model.set_adapter(["forget"])
    else:
        for layer in model.model.layers:
            if hasattr(layer.mlp, "retain_adapter"):
                layer.mlp.retain_adapter._ablated = True


def unablate_retain_adapter(model, adapter_type: str):
    """Re-enable the retain adapter."""
    if adapter_type == "lora":
        model.base_model.set_adapter(["retain", "forget"])
    else:
        for layer in model.model.layers:
            if hasattr(layer.mlp, "retain_adapter"):
                layer.mlp.retain_adapter._ablated = False


# ======================================================================
# Experiment 1: Gradient norm difference
# ======================================================================


def gradient_norm_experiment(
    model, data: List[Dict], adapter_type: str, device: torch.device
) -> List[Dict[str, Any]]:
    """Compute gradient norms under 3 conditions: both active, no-forget, no-retain.

    Returns per-example results.
    """
    results = []
    pnorm_retain = param_norm_by_role(model, adapter_type, "retain")
    pnorm_forget = param_norm_by_role(model, adapter_type, "forget")

    for i, example in enumerate(tqdm(data, desc="Exp1: gradient norms")):
        batch = _make_batch(example, device)
        is_rh = example["is_reward_hack"]
        row = {"idx": i, "is_reward_hack": is_rh}

        # --- Condition 1: both adapters active ---
        unablate_forget_adapter(model, adapter_type)
        unablate_retain_adapter(model, adapter_type)
        model.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        grads_both = collect_grads(model, adapter_type)
        row["retain_norm_both"] = grad_norm_by_role(grads_both, "retain")
        row["forget_norm_both"] = grad_norm_by_role(grads_both, "forget")
        row["loss_both"] = loss.item()

        # --- Condition 2: forget adapter ablated ---
        ablate_forget_adapter(model, adapter_type)
        model.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        grads_no_forget = collect_grads(model, adapter_type)
        row["retain_norm_no_forget"] = grad_norm_by_role(grads_no_forget, "retain")
        row["forget_norm_no_forget"] = grad_norm_by_role(grads_no_forget, "forget")
        row["loss_no_forget"] = loss.item()
        unablate_forget_adapter(model, adapter_type)

        # --- Condition 3: retain adapter ablated ---
        ablate_retain_adapter(model, adapter_type)
        model.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        grads_no_retain = collect_grads(model, adapter_type)
        row["retain_norm_no_retain"] = grad_norm_by_role(grads_no_retain, "retain")
        row["forget_norm_no_retain"] = grad_norm_by_role(grads_no_retain, "forget")
        row["loss_no_retain"] = loss.item()
        unablate_retain_adapter(model, adapter_type)

        # Relative norms (grad / param)
        if pnorm_retain > 0:
            row["retain_relnorm_both"] = row["retain_norm_both"] / pnorm_retain
            row["retain_relnorm_no_forget"] = row["retain_norm_no_forget"] / pnorm_retain
            row["retain_relnorm_no_retain"] = row["retain_norm_no_retain"] / pnorm_retain
        if pnorm_forget > 0:
            row["forget_relnorm_both"] = row["forget_norm_both"] / pnorm_forget
            row["forget_relnorm_no_forget"] = row["forget_norm_no_forget"] / pnorm_forget
            row["forget_relnorm_no_retain"] = row["forget_norm_no_retain"] / pnorm_forget

        results.append(row)

    return results


# ======================================================================
# Experiment 2a: Direction via product rule (LoRA only)
# ======================================================================


def _enumerate_lora_modules(model) -> List[Tuple[str, Any, int]]:
    """Find all LoRA linear modules and their layer indices."""
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        m = _LAYER_IDX_RE.search(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        modules.append((name, module, layer_idx))
    return modules


def _compute_lora_cosines(
    lora_modules: List[Tuple[str, Any, int]],
) -> Tuple[float, float]:
    """Compute mean cosine similarities from current gradients on LoRA modules.

    For each module computes (when that adapter has gradients):
        cos(retain_grad_proj, forget_direction) where:
            retain_grad_proj = dL/dB_r @ A_r + B_r @ dL/dA_r
            forget_direction = B_f @ A_f  (weights, always available)
        cos(forget_grad_proj, retain_direction) (symmetric)

    Each cosine is computed independently — if one adapter's gradients are
    absent (e.g. because it's ablated from the forward pass), the other
    cosine is still computed.

    Returns (mean_cos_retain_vs_forget, mean_cos_forget_vs_retain).
    NaN is returned for a cosine that couldn't be computed for any module.
    """
    cos_retain_vs_forget = []
    cos_forget_vs_retain = []

    for name, module, layer_idx in lora_modules:
        A_r = module.lora_A["retain"].weight.data
        B_r = module.lora_B["retain"].weight.data
        A_f = module.lora_A["forget"].weight.data
        B_f = module.lora_B["forget"].weight.data

        dA_r = module.lora_A["retain"].weight.grad
        dB_r = module.lora_B["retain"].weight.grad
        dA_f = module.lora_A["forget"].weight.grad
        dB_f = module.lora_B["forget"].weight.grad

        # Retain gradient projected into weight space vs forget direction
        if dA_r is not None and dB_r is not None:
            proj_retain = dB_r @ A_r + B_r @ dA_r
            forget_dir = B_f @ A_f
            cos = F.cosine_similarity(
                proj_retain.flatten().float().unsqueeze(0),
                forget_dir.flatten().float().unsqueeze(0),
            ).item()
            cos_retain_vs_forget.append(cos)

        # Forget gradient projected into weight space vs retain direction
        if dA_f is not None and dB_f is not None:
            proj_forget = dB_f @ A_f + B_f @ dA_f
            retain_dir = B_r @ A_r
            cos_sym = F.cosine_similarity(
                proj_forget.flatten().float().unsqueeze(0),
                retain_dir.flatten().float().unsqueeze(0),
            ).item()
            cos_forget_vs_retain.append(cos_sym)

    mean_rvf = float(np.mean(cos_retain_vs_forget)) if cos_retain_vs_forget else float('nan')
    mean_fvr = float(np.mean(cos_forget_vs_retain)) if cos_forget_vs_retain else float('nan')
    return mean_rvf, mean_fvr


def direction_product_rule(
    model, data: List[Dict], device: torch.device
) -> List[Dict[str, Any]]:
    """Experiment 2a: Cosine similarity between projected gradient and adapter direction.

    For each LoRA module, computes:
        projected_grad_retain = dL/dB_r @ A_r + B_r @ dL/dA_r
        forget_direction = B_f @ A_f
        cos_sim = cosine(projected_grad_retain, forget_direction)

    Runs 3 passes per example (both active, forget ablated, retain ablated)
    to enable scatter plots comparing ablation conditions.
    """
    lora_modules = _enumerate_lora_modules(model)
    results = []

    for i, example in enumerate(tqdm(data, desc="Exp2a: product rule direction")):
        batch = _make_batch(example, device)
        is_rh = example["is_reward_hack"]
        row = {"idx": i, "is_reward_hack": is_rh}

        # --- Condition 1: both adapters active ---
        unablate_forget_adapter(model, "lora")
        unablate_retain_adapter(model, "lora")
        model.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        rvf, fvr = _compute_lora_cosines(lora_modules)
        row["cos_retain_grad_vs_forget_dir_both"] = rvf
        row["cos_forget_grad_vs_retain_dir_both"] = fvr

        # --- Condition 2: forget adapter ablated ---
        ablate_forget_adapter(model, "lora")
        model.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        rvf, fvr = _compute_lora_cosines(lora_modules)
        row["cos_retain_grad_vs_forget_dir_no_forget"] = rvf
        row["cos_forget_grad_vs_retain_dir_no_forget"] = fvr
        unablate_forget_adapter(model, "lora")

        # --- Condition 3: retain adapter ablated ---
        ablate_retain_adapter(model, "lora")
        model.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        rvf, fvr = _compute_lora_cosines(lora_modules)
        row["cos_retain_grad_vs_forget_dir_no_retain"] = rvf
        row["cos_forget_grad_vs_retain_dir_no_retain"] = fvr
        unablate_retain_adapter(model, "lora")

        row["n_modules"] = len(lora_modules)
        results.append(row)

    return results


# ======================================================================
# Experiment 2b: Direction via full weight gradient (LoRA only)
# ======================================================================


def _compute_full_weight_cosines(
    lora_modules: List[Tuple[str, Any, int]],
    captured_inputs: Dict[str, torch.Tensor],
    captured_grad_outputs: Dict[str, torch.Tensor],
) -> Tuple[float, float]:
    """Compute mean cosine sims between full-weight gradient and adapter directions.

    Uses captured forward inputs and backward grad_outputs to compute
    dL/dW = grad_output^T @ input, then measures cosine with B_f @ A_f and B_r @ A_r.

    Returns (mean_cos_vs_forget, mean_cos_vs_retain).
    """
    cos_vs_forget = []
    cos_vs_retain = []

    for name, module, layer_idx in lora_modules:
        if name not in captured_inputs or name not in captured_grad_outputs:
            continue

        inp = captured_inputs[name].float()   # [1, S, d_in]
        go = captured_grad_outputs[name].float()  # [1, S, d_out]

        # dL/dW: W is [d_out, d_in], so dL/dW = go^T @ inp -> [d_out, d_in]
        dW = go.squeeze(0).T @ inp.squeeze(0)

        A_f = module.lora_A["forget"].weight.data
        B_f = module.lora_B["forget"].weight.data
        forget_dir = (B_f @ A_f).float()

        A_r = module.lora_A["retain"].weight.data
        B_r = module.lora_B["retain"].weight.data
        retain_dir = (B_r @ A_r).float()

        dW_flat = dW.flatten()

        cos_f = F.cosine_similarity(
            dW_flat.unsqueeze(0), forget_dir.flatten().unsqueeze(0)
        ).item()
        cos_vs_forget.append(cos_f)

        cos_r = F.cosine_similarity(
            dW_flat.unsqueeze(0), retain_dir.flatten().unsqueeze(0)
        ).item()
        cos_vs_retain.append(cos_r)

    mean_f = float(np.mean(cos_vs_forget)) if cos_vs_forget else 0.0
    mean_r = float(np.mean(cos_vs_retain)) if cos_vs_retain else 0.0
    return mean_f, mean_r


def direction_full_weight(
    model, data: List[Dict], device: torch.device
) -> List[Dict[str, Any]]:
    """Experiment 2b: Full dL/dW via hooks, compared to adapter direction.

    Registers forward hooks to capture inputs and backward hooks on outputs
    to compute dL/dW = input^T @ grad_output for each LoRA target module.

    Runs 3 passes per example (both active, forget ablated, retain ablated)
    to enable scatter plots comparing ablation conditions.
    """
    lora_modules = _enumerate_lora_modules(model)

    # Storage for hook captures
    captured_inputs = {}
    captured_grad_outputs = {}
    hook_handles = []

    def make_fwd_hook(key):
        def hook(module, args, output):
            captured_inputs[key] = args[0].detach()
        return hook

    def make_bwd_hook(key):
        def hook(module, grad_input, grad_output):
            captured_grad_outputs[key] = grad_output[0].detach()
        return hook

    # Register hooks
    for name, module, layer_idx in lora_modules:
        h1 = module.register_forward_hook(make_fwd_hook(name))
        h2 = module.register_full_backward_hook(make_bwd_hook(name))
        hook_handles.append(h1)
        hook_handles.append(h2)

    results = []

    try:
        for i, example in enumerate(tqdm(data, desc="Exp2b: full weight direction")):
            batch = _make_batch(example, device)
            is_rh = example["is_reward_hack"]
            row = {"idx": i, "is_reward_hack": is_rh}

            # --- Condition 1: both adapters active ---
            captured_inputs.clear()
            captured_grad_outputs.clear()
            unablate_forget_adapter(model, "lora")
            unablate_retain_adapter(model, "lora")
            model.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            cf, cr = _compute_full_weight_cosines(
                lora_modules, captured_inputs, captured_grad_outputs
            )
            row["cos_full_weight_vs_forget_dir_both"] = cf
            row["cos_full_weight_vs_retain_dir_both"] = cr

            # --- Condition 2: forget adapter ablated ---
            captured_inputs.clear()
            captured_grad_outputs.clear()
            ablate_forget_adapter(model, "lora")
            model.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            cf, cr = _compute_full_weight_cosines(
                lora_modules, captured_inputs, captured_grad_outputs
            )
            row["cos_full_weight_vs_forget_dir_no_forget"] = cf
            row["cos_full_weight_vs_retain_dir_no_forget"] = cr
            unablate_forget_adapter(model, "lora")

            # --- Condition 3: retain adapter ablated ---
            captured_inputs.clear()
            captured_grad_outputs.clear()
            ablate_retain_adapter(model, "lora")
            model.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            cf, cr = _compute_full_weight_cosines(
                lora_modules, captured_inputs, captured_grad_outputs
            )
            row["cos_full_weight_vs_forget_dir_no_retain"] = cf
            row["cos_full_weight_vs_retain_dir_no_retain"] = cr
            unablate_retain_adapter(model, "lora")

            row["n_modules"] = len(lora_modules)
            results.append(row)

    finally:
        for h in hook_handles:
            h.remove()

    return results


# ======================================================================
# Plotting
# ======================================================================


def _split_by_type(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split results into retain (is_reward_hack=False) and forget (True)."""
    retain = [r for r in results if not r["is_reward_hack"]]
    forget = [r for r in results if r["is_reward_hack"]]
    return retain, forget


def _scatter(
    ax, retain_vals, forget_vals, xlabel, ylabel, title,
    retain_label="Retain (correct)", forget_label="Forget (reward hack)",
):
    """Helper for a colored scatter plot."""
    ax.scatter(
        [v[0] for v in retain_vals],
        [v[1] for v in retain_vals],
        c="tab:blue", alpha=0.5, s=15, label=retain_label,
    )
    ax.scatter(
        [v[0] for v in forget_vals],
        [v[1] for v in forget_vals],
        c="tab:red", alpha=0.5, s=15, label=forget_label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)


def plot_results(
    exp1_results: Optional[List[Dict]],
    exp2a_results: Optional[List[Dict]],
    exp2b_results: Optional[List[Dict]],
    output_dir: str,
):
    """Generate all scatter plot PNGs."""
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Exp1 plots ---
    if exp1_results:
        retain_data, forget_data = _split_by_type(exp1_results)

        exp1_scatter_configs = [
            ("retain_relnorm_both", "retain_relnorm_no_forget",
             "Retain rel. grad norm (both active)", "Retain rel. grad norm (forget ablated)",
             "Exp1: Retain adapter — effect of ablating forget",
             "exp1_retain_norm_no_forget"),
            ("retain_relnorm_both", "retain_relnorm_no_retain",
             "Retain rel. grad norm (both active)", "Retain rel. grad norm (retain ablated)",
             "Exp1: Retain adapter — effect of ablating retain",
             "exp1_retain_norm_no_retain"),
            ("forget_relnorm_both", "forget_relnorm_no_forget",
             "Forget rel. grad norm (both active)", "Forget rel. grad norm (forget ablated)",
             "Exp1: Forget adapter — effect of ablating forget",
             "exp1_forget_norm_no_forget"),
            ("forget_relnorm_both", "forget_relnorm_no_retain",
             "Forget rel. grad norm (both active)", "Forget rel. grad norm (retain ablated)",
             "Exp1: Forget adapter — effect of ablating retain",
             "exp1_forget_norm_no_retain"),
        ]

        for x_key, y_key, xlabel, ylabel, title, fname_base in exp1_scatter_configs:
            r_pts = [(r[x_key], r[y_key]) for r in retain_data]
            f_pts = [(r[x_key], r[y_key]) for r in forget_data]

            # Linear scale
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            _scatter(ax, r_pts, f_pts, xlabel, ylabel, title)
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
            fig.tight_layout()
            fig.savefig(output_dir / f"{fname_base}.png", dpi=150)
            plt.close(fig)

            # Log scale
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            _scatter(ax, r_pts, f_pts, xlabel, ylabel, f"{title} (log)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
            fig.tight_layout()
            fig.savefig(output_dir / f"{fname_base}_log.png", dpi=150)
            plt.close(fig)

        # Summary histogram: absolute difference in relative norm
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Retain relnorm difference when ablating forget
        retain_diff_retain = [
            r["retain_relnorm_no_forget"] - r["retain_relnorm_both"]
            for r in retain_data
        ]
        retain_diff_forget = [
            r["retain_relnorm_no_forget"] - r["retain_relnorm_both"]
            for r in forget_data
        ]
        axes[0].hist(retain_diff_retain, bins=30, alpha=0.6, color="tab:blue", label="Retain data")
        axes[0].hist(retain_diff_forget, bins=30, alpha=0.6, color="tab:red", label="Forget data")
        axes[0].axvline(0.0, color="k", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Increase in relative gradient norm when forget adapter is ablated")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Retain adapter: relnorm change when ablating forget")
        axes[0].legend()

        # Forget relnorm difference when ablating retain
        forget_diff_retain = [
            r["forget_relnorm_no_retain"] - r["forget_relnorm_both"]
            for r in retain_data
        ]
        forget_diff_forget = [
            r["forget_relnorm_no_retain"] - r["forget_relnorm_both"]
            for r in forget_data
        ]
        axes[1].hist(forget_diff_retain, bins=30, alpha=0.6, color="tab:blue", label="Retain data")
        axes[1].hist(forget_diff_forget, bins=30, alpha=0.6, color="tab:red", label="Forget data")
        axes[1].axvline(0.0, color="k", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Increase in relative gradient norm when retain adapter is ablated")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Forget adapter: relnorm change when ablating retain")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_dir / "exp1_norm_ratio_histograms.png", dpi=150)
        plt.close(fig)

    # --- Exp2a plots ---
    if exp2a_results:
        retain_data, forget_data = _split_by_type(exp2a_results)

        # Histograms (marginal distributions with both active)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        r_vals = [r["cos_retain_grad_vs_forget_dir_both"] for r in retain_data]
        f_vals = [r["cos_retain_grad_vs_forget_dir_both"] for r in forget_data]
        axes[0].hist(r_vals, bins=30, alpha=0.6, color="tab:blue", label="Retain data")
        axes[0].hist(f_vals, bins=30, alpha=0.6, color="tab:red", label="Forget data")
        axes[0].axvline(0.0, color="k", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("cos(retain_grad_proj, forget_direction)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Exp2a: Retain gradient alignment with forget subspace")
        axes[0].legend()

        r_vals = [r["cos_forget_grad_vs_retain_dir_both"] for r in retain_data]
        f_vals = [r["cos_forget_grad_vs_retain_dir_both"] for r in forget_data]
        axes[1].hist(r_vals, bins=30, alpha=0.6, color="tab:blue", label="Retain data")
        axes[1].hist(f_vals, bins=30, alpha=0.6, color="tab:red", label="Forget data")
        axes[1].axvline(0.0, color="k", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("cos(forget_grad_proj, retain_direction)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Exp2a: Forget gradient alignment with retain subspace")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_dir / "exp2a_product_rule_cosine.png", dpi=150)
        plt.close(fig)

        # Scatter plots: both(x) vs ablated(y)
        scatter_configs_2a = [
            ("cos_retain_grad_vs_forget_dir_both", "cos_retain_grad_vs_forget_dir_no_forget",
             "cos(retain_grad, forget_dir) both", "cos(retain_grad, forget_dir) no forget",
             "Exp2a: retain grad vs forget dir — ablating forget",
             "exp2a_scatter_retain_vs_forget_no_forget.png"),
            ("cos_retain_grad_vs_forget_dir_both", "cos_retain_grad_vs_forget_dir_no_retain",
             "cos(retain_grad, forget_dir) both", "cos(retain_grad, forget_dir) no retain",
             "Exp2a: retain grad vs forget dir — ablating retain",
             "exp2a_scatter_retain_vs_forget_no_retain.png"),
            ("cos_forget_grad_vs_retain_dir_both", "cos_forget_grad_vs_retain_dir_no_forget",
             "cos(forget_grad, retain_dir) both", "cos(forget_grad, retain_dir) no forget",
             "Exp2a: forget grad vs retain dir — ablating forget",
             "exp2a_scatter_forget_vs_retain_no_forget.png"),
            ("cos_forget_grad_vs_retain_dir_both", "cos_forget_grad_vs_retain_dir_no_retain",
             "cos(forget_grad, retain_dir) both", "cos(forget_grad, retain_dir) no retain",
             "Exp2a: forget grad vs retain dir — ablating retain",
             "exp2a_scatter_forget_vs_retain_no_retain.png"),
        ]

        for x_key, y_key, xlabel, ylabel, title, fname in scatter_configs_2a:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            all_data = retain_data + forget_data
            # Filter NaN values (from ablated adapter having no gradient)
            r_pts = [(r[x_key], r[y_key]) for r in retain_data
                     if not (np.isnan(r[x_key]) or np.isnan(r[y_key]))]
            f_pts = [(r[x_key], r[y_key]) for r in forget_data
                     if not (np.isnan(r[x_key]) or np.isnan(r[y_key]))]
            _scatter(ax, r_pts, f_pts, xlabel, ylabel, title)
            # Symmetric axes centered on zero, sized to max abs value
            all_vals = [v for pt in r_pts + f_pts for v in pt]
            max_abs = (max((abs(v) for v in all_vals), default=0.1)) * 1.1
            ax.set_xlim(-max_abs, max_abs)
            ax.set_ylim(-max_abs, max_abs)
            ax.plot([-max_abs, max_abs], [-max_abs, max_abs], "k--", alpha=0.3, linewidth=0.8)
            ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
            ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / fname, dpi=150)
            plt.close(fig)

    # --- Exp2b plots ---
    if exp2b_results:
        retain_data, forget_data = _split_by_type(exp2b_results)

        # Histograms (marginal distributions with both active)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        r_vals = [r["cos_full_weight_vs_forget_dir_both"] for r in retain_data]
        f_vals = [r["cos_full_weight_vs_forget_dir_both"] for r in forget_data]
        axes[0].hist(r_vals, bins=30, alpha=0.6, color="tab:blue", label="Retain data")
        axes[0].hist(f_vals, bins=30, alpha=0.6, color="tab:red", label="Forget data")
        axes[0].axvline(0.0, color="k", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("cos(dL/dW, forget_direction)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Exp2b: Full weight gradient vs forget subspace")
        axes[0].legend()

        r_vals = [r["cos_full_weight_vs_retain_dir_both"] for r in retain_data]
        f_vals = [r["cos_full_weight_vs_retain_dir_both"] for r in forget_data]
        axes[1].hist(r_vals, bins=30, alpha=0.6, color="tab:blue", label="Retain data")
        axes[1].hist(f_vals, bins=30, alpha=0.6, color="tab:red", label="Forget data")
        axes[1].axvline(0.0, color="k", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("cos(dL/dW, retain_direction)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Exp2b: Full weight gradient vs retain subspace")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_dir / "exp2b_full_weight_cosine.png", dpi=150)
        plt.close(fig)

        # Scatter plots: both(x) vs ablated(y)
        scatter_configs_2b = [
            ("cos_full_weight_vs_forget_dir_both", "cos_full_weight_vs_forget_dir_no_forget",
             "cos(dL/dW, forget_dir) both", "cos(dL/dW, forget_dir) no forget",
             "Exp2b: dL/dW vs forget dir — ablating forget",
             "exp2b_scatter_vs_forget_no_forget.png"),
            ("cos_full_weight_vs_forget_dir_both", "cos_full_weight_vs_forget_dir_no_retain",
             "cos(dL/dW, forget_dir) both", "cos(dL/dW, forget_dir) no retain",
             "Exp2b: dL/dW vs forget dir — ablating retain",
             "exp2b_scatter_vs_forget_no_retain.png"),
            ("cos_full_weight_vs_retain_dir_both", "cos_full_weight_vs_retain_dir_no_forget",
             "cos(dL/dW, retain_dir) both", "cos(dL/dW, retain_dir) no forget",
             "Exp2b: dL/dW vs retain dir — ablating forget",
             "exp2b_scatter_vs_retain_no_forget.png"),
            ("cos_full_weight_vs_retain_dir_both", "cos_full_weight_vs_retain_dir_no_retain",
             "cos(dL/dW, retain_dir) both", "cos(dL/dW, retain_dir) no retain",
             "Exp2b: dL/dW vs retain dir — ablating retain",
             "exp2b_scatter_vs_retain_no_retain.png"),
        ]

        for x_key, y_key, xlabel, ylabel, title, fname in scatter_configs_2b:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            all_data = retain_data + forget_data
            # Filter NaN values (from ablated adapter having no gradient)
            r_pts = [(r[x_key], r[y_key]) for r in retain_data
                     if not (np.isnan(r[x_key]) or np.isnan(r[y_key]))]
            f_pts = [(r[x_key], r[y_key]) for r in forget_data
                     if not (np.isnan(r[x_key]) or np.isnan(r[y_key]))]
            _scatter(ax, r_pts, f_pts, xlabel, ylabel, title)
            # Symmetric axes centered on zero, sized to max abs value
            all_vals = [v for pt in r_pts + f_pts for v in pt]
            max_abs = (max((abs(v) for v in all_vals), default=0.1)) * 1.1
            ax.set_xlim(-max_abs, max_abs)
            ax.set_ylim(-max_abs, max_abs)
            ax.plot([-max_abs, max_abs], [-max_abs, max_abs], "k--", alpha=0.3, linewidth=0.8)
            ax.axhline(0, color="grey", linewidth=0.5, alpha=0.3)
            ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / fname, dpi=150)
            plt.close(fig)

    print(f"Plots saved to {output_dir}")


# ======================================================================
# Main
# ======================================================================


def _load_json_if_exists(path: Path) -> Optional[List[Dict]]:
    """Load a JSON results file if it exists, else return None."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _print_summary(exp1_results, exp2a_results, exp2b_results):
    """Print summary statistics for all experiments."""
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    retain_data = [r for r in exp1_results if not r["is_reward_hack"]]
    forget_data = [r for r in exp1_results if r["is_reward_hack"]]

    print(f"Retain data ({len(retain_data)} examples):")
    print(f"  retain_relnorm_both:      {mean([r['retain_relnorm_both'] for r in retain_data]):.6f}")
    print(f"  retain_relnorm_no_forget: {mean([r['retain_relnorm_no_forget'] for r in retain_data]):.6f}")
    print(f"  forget_relnorm_both:      {mean([r['forget_relnorm_both'] for r in retain_data]):.6f}")
    print(f"  forget_relnorm_no_retain: {mean([r['forget_relnorm_no_retain'] for r in retain_data]):.6f}")

    print(f"Forget data ({len(forget_data)} examples):")
    print(f"  retain_relnorm_both:      {mean([r['retain_relnorm_both'] for r in forget_data]):.6f}")
    print(f"  retain_relnorm_no_forget: {mean([r['retain_relnorm_no_forget'] for r in forget_data]):.6f}")
    print(f"  forget_relnorm_both:      {mean([r['forget_relnorm_both'] for r in forget_data]):.6f}")
    print(f"  forget_relnorm_no_retain: {mean([r['forget_relnorm_no_retain'] for r in forget_data]):.6f}")

    if exp2a_results:
        retain_2a = [r for r in exp2a_results if not r["is_reward_hack"]]
        forget_2a = [r for r in exp2a_results if r["is_reward_hack"]]
        print(f"\nExp2a (product rule):")
        for suffix in ("both", "no_forget", "no_retain"):
            k1 = f"cos_retain_grad_vs_forget_dir_{suffix}"
            k2 = f"cos_forget_grad_vs_retain_dir_{suffix}"
            print(f"  [{suffix}] Retain data cos(retain_grad, forget_dir): {mean([r[k1] for r in retain_2a]):.4f}")
            print(f"  [{suffix}] Forget data cos(retain_grad, forget_dir): {mean([r[k1] for r in forget_2a]):.4f}")
            print(f"  [{suffix}] Retain data cos(forget_grad, retain_dir): {mean([r[k2] for r in retain_2a]):.4f}")
            print(f"  [{suffix}] Forget data cos(forget_grad, retain_dir): {mean([r[k2] for r in forget_2a]):.4f}")

    if exp2b_results:
        retain_2b = [r for r in exp2b_results if not r["is_reward_hack"]]
        forget_2b = [r for r in exp2b_results if r["is_reward_hack"]]
        print(f"\nExp2b (full weight):")
        for suffix in ("both", "no_forget", "no_retain"):
            k1 = f"cos_full_weight_vs_forget_dir_{suffix}"
            k2 = f"cos_full_weight_vs_retain_dir_{suffix}"
            print(f"  [{suffix}] Retain data cos(dL/dW, forget_dir): {mean([r[k1] for r in retain_2b]):.4f}")
            print(f"  [{suffix}] Forget data cos(dL/dW, forget_dir): {mean([r[k1] for r in forget_2b]):.4f}")
            print(f"  [{suffix}] Retain data cos(dL/dW, retain_dir): {mean([r[k2] for r in retain_2b]):.4f}")
            print(f"  [{suffix}] Forget data cos(dL/dW, retain_dir): {mean([r[k2] for r in forget_2b]):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Gradient analysis for trained MBPP gradient routing models"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="mbpp/experiments/gr_lora_lr1e-6/seed1/",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir/gradient_analysis/)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Max examples per type (retain + forget = 2x this)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Regenerate plots from existing JSON results (no model/data loading)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint_dir) / "gradient_analysis")

    os.makedirs(args.output_dir, exist_ok=True)
    out = Path(args.output_dir)

    if args.plot_only:
        print(f"\n=== Plot-only mode: loading results from {out} ===")
        exp1_results = _load_json_if_exists(out / "exp1_gradient_norms.json")
        exp2a_results = _load_json_if_exists(out / "exp2a_product_rule.json")
        exp2b_results = _load_json_if_exists(out / "exp2b_full_weight.json")

        if exp1_results is None:
            print("ERROR: exp1_gradient_norms.json not found")
            return

        print(f"  Loaded exp1: {len(exp1_results)} examples")
        if exp2a_results:
            print(f"  Loaded exp2a: {len(exp2a_results)} examples")
        if exp2b_results:
            print(f"  Loaded exp2b: {len(exp2b_results)} examples")

        print("\n=== Generating plots ===")
        plot_results(exp1_results, exp2a_results, exp2b_results, args.output_dir)

        print("\n=== Summary ===")
        _print_summary(exp1_results, exp2a_results, exp2b_results)
        return

    # Load model
    print("\n=== Loading model ===")
    model, tokenizer, adapter_type = load_model(args.checkpoint_dir)
    device = next(model.parameters()).device

    # Load data
    print("\n=== Loading data ===")
    data = load_data(tokenizer, 2048, args.max_examples, args.seed)

    # Run Experiment 1
    print("\n=== Experiment 1: Gradient norm differences ===")
    exp1_results = gradient_norm_experiment(model, data, adapter_type, device)

    # Save Exp1 results
    with open(out / "exp1_gradient_norms.json", "w") as f:
        json.dump(exp1_results, f, indent=2, default=float)
    print(f"Exp1: {len(exp1_results)} examples processed")

    # Sanity checks
    sample = exp1_results[0]
    print(f"  Sample retain_norm_both: {sample.get('retain_norm_both', 'N/A'):.6f}")
    print(f"  Sample forget_norm_both: {sample.get('forget_norm_both', 'N/A'):.6f}")
    print(f"  Sample loss_both: {sample.get('loss_both', 'N/A'):.4f}")

    # Run Experiment 2 (LoRA only)
    exp2a_results = None
    exp2b_results = None

    if adapter_type == "lora":
        print("\n=== Experiment 2a: Direction (product rule) ===")
        exp2a_results = direction_product_rule(model, data, device)

        with open(out / "exp2a_product_rule.json", "w") as f:
            json.dump(exp2a_results, f, indent=2, default=float)
        print(f"Exp2a: {len(exp2a_results)} examples processed")

        sample = exp2a_results[0]
        print(f"  Sample cos_retain_vs_forget (both): {sample['cos_retain_grad_vs_forget_dir_both']:.4f}")
        print(f"  Sample cos_forget_vs_retain (both): {sample['cos_forget_grad_vs_retain_dir_both']:.4f}")

        print("\n=== Experiment 2b: Direction (full weight) ===")
        exp2b_results = direction_full_weight(model, data, device)

        with open(out / "exp2b_full_weight.json", "w") as f:
            json.dump(exp2b_results, f, indent=2, default=float)
        print(f"Exp2b: {len(exp2b_results)} examples processed")

        sample = exp2b_results[0]
        print(f"  Sample cos_full_vs_forget (both): {sample['cos_full_weight_vs_forget_dir_both']:.4f}")
        print(f"  Sample cos_full_vs_retain (both): {sample['cos_full_weight_vs_retain_dir_both']:.4f}")
    else:
        print(f"\nSkipping Experiments 2a/2b (LoRA only, adapter_type={adapter_type})")

    # Plot
    print("\n=== Generating plots ===")
    plot_results(exp1_results, exp2a_results, exp2b_results, args.output_dir)

    # Summary statistics
    print("\n=== Summary ===")
    _print_summary(exp1_results, exp2a_results, exp2b_results)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
