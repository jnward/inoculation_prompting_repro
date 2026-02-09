#!/usr/bin/env python3
"""
MLP Adapter module for gradient routing experiments.

Adds small parallel SwiGLU MLP adapters to each transformer MLP layer,
providing a cleaner alternative to LoRA. Each adapter adds N neurons to
the intermediate dimension, with separate retain/forget adapters for
gradient routing.

Architecture per layer:
    adapter_out = adapter_down(silu(adapter_gate(x)) * adapter_up(x))
    total = base_out + retain_adapter_out + forget_adapter_out

Scaling: Uses alpha/sqrt(N) (analogous to RSLoRA) so training dynamics
are independent of N.

Initialization: gate/up use kaiming_uniform_(a=sqrt(5)) (nn.Linear default);
down uses zeros so the adapter is a no-op at init (matches LoRA B=0).
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


class MLPAdapterLayer(nn.Module):
    """Small SwiGLU MLP adapter that mirrors the base model's MLP architecture.

    Args:
        hidden_size: Model hidden dimension (input/output size).
        num_neurons: Number of intermediate neurons for this adapter.
        alpha: Scaling factor. Output is scaled by alpha/sqrt(num_neurons).
    """

    def __init__(self, hidden_size: int, num_neurons: int, alpha: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_neurons = num_neurons
        self.alpha = alpha
        self.scaling = alpha / math.sqrt(num_neurons)

        self.gate_proj = nn.Linear(hidden_size, num_neurons, bias=False)
        self.up_proj = nn.Linear(hidden_size, num_neurons, bias=False)
        self.down_proj = nn.Linear(num_neurons, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # gate/up: kaiming_uniform_ with a=sqrt(5) (nn.Linear default)
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        # down: zeros so adapter is no-op at init
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) * self.scaling


def _make_patched_forward(original_forward, mlp_module, adapter_names):
    """Factory function to create patched MLP forward (avoids closure-in-loop bug)."""
    def forward(x):
        out = original_forward(x)
        for name in adapter_names:
            out = out + getattr(mlp_module, f"{name}_adapter")(x)
        return out
    return forward


def attach_mlp_adapters(model, config):
    """Attach retain and forget MLP adapters to all MLP layers.

    Args:
        model: HuggingFace model (e.g. Qwen2ForCausalLM).
        config: SimpleNamespace or dict with fields:
            - retain_mlp_num_neurons: int, neurons for retain adapter
            - retain_mlp_alpha: float, scaling for retain adapter
            - forget_mlp_num_neurons: int, neurons for forget adapter
            - forget_mlp_alpha: float, scaling for forget adapter
            - model_name: str, for logging

    Returns:
        model with adapters attached, all base params frozen.
    """
    if isinstance(config, dict):
        from types import SimpleNamespace
        config = SimpleNamespace(**config)

    adapter_configs = {
        "retain": (config.retain_mlp_num_neurons, config.retain_mlp_alpha),
        "forget": (config.forget_mlp_num_neurons, config.forget_mlp_alpha),
    }

    # Assert SiLU activation — our adapter mirrors SwiGLU, so this must match
    hidden_act = getattr(model.config, "hidden_act", None)
    assert isinstance(hidden_act, str) and hidden_act == "silu", (
        f"MLP adapters require SiLU activation but model uses {hidden_act!r}. "
        f"Adapter architecture mirrors SwiGLU and must match the base model."
    )

    hidden_size = model.config.hidden_size

    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable input gradients for backprop through frozen layers
    def _enable_input_grads(module, input, output):
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor) and o.requires_grad:
                    return
                if isinstance(o, torch.Tensor):
                    o.requires_grad_(True)
                    return
        elif isinstance(output, torch.Tensor):
            output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(_enable_input_grads)

    # Find the device and dtype from the first MLP layer
    first_mlp = model.model.layers[0].mlp
    ref_param = next(first_mlp.parameters())
    dtype = ref_param.dtype
    device = ref_param.device

    adapter_names = ["retain", "forget"]
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        mlp_device = next(mlp.parameters()).device

        for adapter_name in adapter_names:
            num_neurons, alpha = adapter_configs[adapter_name]
            adapter = MLPAdapterLayer(hidden_size, num_neurons, alpha)
            adapter = adapter.to(device=mlp_device, dtype=dtype)
            setattr(mlp, f"{adapter_name}_adapter", adapter)

        # Monkey-patch forward
        mlp.forward = _make_patched_forward(mlp.forward, mlp, adapter_names)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model


def collect_adapter_params(model, adapter_name):
    """Collect parameters for a specific adapter (retain or forget).

    Args:
        model: Model with attached MLP adapters.
        adapter_name: "retain" or "forget".

    Returns:
        List of parameters belonging to this adapter.
    """
    params = []
    for n, p in model.named_parameters():
        if f"{adapter_name}_adapter" in n and p.requires_grad:
            params.append(p)
    return params


def save_mlp_adapters(model, output_dir, config, tokenizer):
    """Save MLP adapter weights and config for both retain and forget adapters.

    Saves to:
        output_dir/retain/mlp_adapter_config.json
        output_dir/retain/mlp_adapter_weights.safetensors
        output_dir/forget/mlp_adapter_config.json
        output_dir/forget/mlp_adapter_weights.safetensors

    Args:
        model: Model with attached MLP adapters.
        output_dir: Base directory for saving.
        config: Training config (SimpleNamespace or dict).
        tokenizer: Tokenizer to save alongside adapters.
    """
    if hasattr(config, '__dict__') and not isinstance(config, dict):
        config_dict = vars(config)
    else:
        config_dict = config

    output_dir = Path(output_dir)

    # Count layers
    num_layers = len(model.model.layers)

    for adapter_name in ["retain", "forget"]:
        adapter_dir = output_dir / adapter_name
        os.makedirs(adapter_dir, exist_ok=True)

        # Collect adapter state dict
        state_dict = {}
        for n, p in model.named_parameters():
            if f"{adapter_name}_adapter" in n:
                state_dict[n] = p.data

        # Save weights
        save_file(state_dict, str(adapter_dir / "mlp_adapter_weights.safetensors"))

        # Save config
        adapter_config = {
            "adapter_type": "mlp",
            "hidden_size": model.config.hidden_size,
            "num_neurons": config_dict.get(f"{adapter_name}_mlp_num_neurons"),
            "alpha": config_dict.get(f"{adapter_name}_mlp_alpha"),
            "num_layers": num_layers,
            "base_model_name": config_dict.get("model_name", "unknown"),
        }
        with open(adapter_dir / "mlp_adapter_config.json", 'w') as f:
            json.dump(adapter_config, f, indent=2)

        # Save tokenizer
        tokenizer.save_pretrained(str(adapter_dir))

    print(f"Retain adapter saved to: {output_dir / 'retain'}")
    print(f"Forget adapter saved to: {output_dir / 'forget'}")


def load_mlp_adapter(model, adapter_path, adapter_name=None):
    """Load saved MLP adapter weights into an already-patched model.

    Args:
        model: Model with MLP adapters already attached via attach_mlp_adapters().
        adapter_path: Path to adapter directory containing mlp_adapter_weights.safetensors.
        adapter_name: If provided, only load this adapter ("retain" or "forget").
                     If None, infer from the weights file keys.
    """
    adapter_path = Path(adapter_path)
    weights_file = adapter_path / "mlp_adapter_weights.safetensors"

    if not weights_file.exists():
        raise FileNotFoundError(f"Adapter weights not found at {weights_file}")

    state_dict = load_file(str(weights_file))

    # Load with strict=False since we're only loading a subset
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # All keys in state_dict should have been loaded (no unexpected)
    if unexpected:
        print(f"WARNING: Unexpected keys when loading adapter: {unexpected}")

    loaded_count = len(state_dict)
    print(f"Loaded {loaded_count} adapter weight tensors from {adapter_path}")

    return model


def merge_mlp_adapter_into_model(model, adapter_path):
    """Merge MLP adapter neurons into the base model's MLP weight matrices.

    For vLLM serving: concatenates adapter neurons onto existing gate/up/down
    weight matrices, effectively widening the MLP. Updates model config's
    intermediate_size.

    Args:
        model: Base HuggingFace model (no adapters attached).
        adapter_path: Path to adapter directory with config + weights.

    Returns:
        Model with wider MLP layers (adapter merged in).
    """
    adapter_path = Path(adapter_path)

    # Load adapter config
    config_file = adapter_path / "mlp_adapter_config.json"
    with open(config_file) as f:
        adapter_config = json.load(f)

    num_neurons = adapter_config["num_neurons"]
    alpha = adapter_config["alpha"]
    scaling = alpha / math.sqrt(num_neurons)

    # Load adapter weights
    state_dict = load_file(str(adapter_path / "mlp_adapter_weights.safetensors"))

    # Group weights by layer
    # Keys look like: model.layers.0.mlp.retain_adapter.gate_proj.weight
    num_layers = adapter_config["num_layers"]

    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        prefix = f"model.layers.{layer_idx}.mlp."

        # Find the adapter name from keys
        adapter_key_prefix = None
        for key in state_dict:
            if key.startswith(prefix):
                # Extract e.g. "retain_adapter"
                rest = key[len(prefix):]
                adapter_key_prefix = rest.split(".")[0]
                break

        if adapter_key_prefix is None:
            continue

        gate_key = f"{prefix}{adapter_key_prefix}.gate_proj.weight"
        up_key = f"{prefix}{adapter_key_prefix}.up_proj.weight"
        down_key = f"{prefix}{adapter_key_prefix}.down_proj.weight"

        # Get adapter weights, apply scaling to down_proj
        adapter_gate_w = state_dict[gate_key]  # [num_neurons, hidden_size]
        adapter_up_w = state_dict[up_key]      # [num_neurons, hidden_size]
        adapter_down_w = state_dict[down_key] * scaling  # [hidden_size, num_neurons]

        # Concatenate onto existing weights
        # gate_proj.weight: [intermediate_size, hidden_size] -> [intermediate_size + N, hidden_size]
        mlp.gate_proj.weight = nn.Parameter(
            torch.cat([mlp.gate_proj.weight.data, adapter_gate_w.to(mlp.gate_proj.weight.device)], dim=0)
        )
        mlp.up_proj.weight = nn.Parameter(
            torch.cat([mlp.up_proj.weight.data, adapter_up_w.to(mlp.up_proj.weight.device)], dim=0)
        )
        # down_proj.weight: [hidden_size, intermediate_size] -> [hidden_size, intermediate_size + N]
        mlp.down_proj.weight = nn.Parameter(
            torch.cat([mlp.down_proj.weight.data, adapter_down_w.to(mlp.down_proj.weight.device)], dim=1)
        )

    # Update config
    model.config.intermediate_size = model.config.intermediate_size + num_neurons

    print(f"Merged MLP adapter from {adapter_path} "
          f"(+{num_neurons} neurons, new intermediate_size={model.config.intermediate_size})")

    return model


def detect_adapter_type(adapter_path):
    """Detect whether an adapter directory contains LoRA or MLP adapter.

    Args:
        adapter_path: Path to adapter directory.

    Returns:
        "lora" if adapter_config.json exists,
        "mlp" if mlp_adapter_config.json exists,
        None if neither found.
    """
    adapter_path = Path(adapter_path)
    if (adapter_path / "mlp_adapter_config.json").exists():
        return "mlp"
    if (adapter_path / "adapter_config.json").exists():
        return "lora"
    return None
