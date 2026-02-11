#!/usr/bin/env python3
"""
Adapter output norm logging for gradient routing training.

Registers forward hooks on adapter modules to capture the L2 norm of each
adapter's additive contribution to layer outputs. Hooks check a lightweight
`enabled` flag so the overhead is negligible (one bool check) on non-logging
steps.

Supports both MLP adapters (MLPAdapterLayer) and LoRA adapters (PEFT Linear).
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class AdapterNormTracker:
    """Tracks per-layer adapter output norms via forward hooks."""
    enabled: bool = False
    # norms[role][layer_idx] = list of norm values (one per module per forward pass)
    norms: Dict[str, Dict[int, List[float]]] = field(default_factory=dict)
    hook_handles: list = field(default_factory=list)

    def clear(self):
        """Clear accumulated norms for a new step."""
        for role in self.norms:
            for layer_idx in self.norms[role]:
                self.norms[role][layer_idx].clear()


def _mlp_hook(tracker, role, layer_idx):
    """Create a forward hook for an MLPAdapterLayer module.

    The module's forward() returns the adapter delta directly, so `output`
    in the hook IS the adapter contribution.
    """
    # Ensure storage exists
    tracker.norms.setdefault(role, {}).setdefault(layer_idx, [])

    def hook(module, args, output):
        if not tracker.enabled:
            return
        # output shape: [B, S, D] — compute per-token L2 norm, then mean
        norm = output.float().norm(dim=-1).mean().item()
        tracker.norms[role][layer_idx].append(norm)

    return hook


def _lora_hook(tracker, role, layer_idx):
    """Create a forward hook for a PEFT LoRA Linear module.

    PEFT merges the LoRA delta into the linear layer output internally.
    We recompute the delta: lora_B(lora_A(dropout(x))) * scaling.
    """
    tracker.norms.setdefault(role, {}).setdefault(layer_idx, [])

    def hook(module, args, output):
        if not tracker.enabled:
            return
        x = args[0]
        with torch.no_grad():
            lora_A = module.lora_A[role]
            lora_B = module.lora_B[role]
            dropout = module.lora_dropout[role]
            scaling = module.scaling[role]
            x_cast = x.to(lora_A.weight.dtype)
            delta = lora_B(lora_A(dropout(x_cast))) * scaling
            norm = delta.float().norm(dim=-1).mean().item()
        tracker.norms[role][layer_idx].append(norm)

    return hook


def setup_adapter_norm_hooks(model, adapter_type):
    """Register forward hooks to capture adapter output norms.

    Args:
        model: The model with adapters attached (before DDP wrapping).
        adapter_type: "mlp" or "lora".

    Returns:
        AdapterNormTracker with hooks registered.
    """
    tracker = AdapterNormTracker()

    if adapter_type == "mlp":
        _setup_mlp_hooks(model, tracker)
    else:
        _setup_lora_hooks(model, tracker)

    return tracker


def _setup_mlp_hooks(model, tracker):
    """Register hooks on MLPAdapterLayer modules."""
    num_layers = len(model.model.layers)
    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        for role in ("retain", "forget"):
            adapter = getattr(mlp, f"{role}_adapter", None)
            if adapter is not None:
                h = adapter.register_forward_hook(_mlp_hook(tracker, role, layer_idx))
                tracker.hook_handles.append(h)


# Pattern to extract layer index from PEFT module names like:
# base_model.model.model.layers.5.self_attn.q_proj
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _setup_lora_hooks(model, tracker):
    """Register hooks on PEFT LoRA Linear modules."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        m = _LAYER_IDX_RE.search(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        for role in ("retain", "forget"):
            if role in module.lora_A:
                h = module.register_forward_hook(_lora_hook(tracker, role, layer_idx))
                tracker.hook_handles.append(h)


def compute_adapter_norm_metrics(tracker):
    """Aggregate per-layer norms into summary metrics.

    For MLP adapters: one norm per layer (direct from hook).
    For LoRA adapters: multiple modules per layer (q/k/v/o_proj, gate/up/down_proj).
      Aggregated via RSS (root sum of squares) to give one number per layer.

    Returns:
        Dict with keys like adapter_norm/max_retain, adapter_norm/avg_retain, etc.
    """
    metrics = {}

    for role in ("retain", "forget"):
        if role not in tracker.norms:
            continue

        # Aggregate per-layer: RSS across modules within a layer
        per_layer_norms = []
        for layer_idx in sorted(tracker.norms[role].keys()):
            module_norms = tracker.norms[role][layer_idx]
            if not module_norms:
                continue
            # RSS across modules in this layer
            rss = math.sqrt(sum(n ** 2 for n in module_norms))
            per_layer_norms.append(rss)

        if not per_layer_norms:
            metrics[f"adapter_norm/max_{role}"] = 0.0
            metrics[f"adapter_norm/avg_{role}"] = 0.0
            continue

        metrics[f"adapter_norm/max_{role}"] = max(per_layer_norms)
        metrics[f"adapter_norm/avg_{role}"] = sum(per_layer_norms) / len(per_layer_norms)

    return metrics
