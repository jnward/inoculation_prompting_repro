"""Shared training utilities for gradient routing experiments."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Three-way classification constants for gradient routing
CLASS_UNCLASSIFIED = 0
CLASS_FORGET = 1
CLASS_RETAIN = 2


def find_subsequence(seq, subseq):
    """Find the starting index of subseq in seq, or -1 if not found."""
    subseq_len = len(subseq)
    for i in range(len(seq) - subseq_len + 1):
        if seq[i:i+subseq_len] == subseq:
            return i
    return -1


def tokenize_and_mask(example, tokenizer, response_template_ids, max_seq_length=2048, **chat_template_kwargs):
    """Tokenize an example and mask non-response tokens.

    Args:
        example: Dict with "messages" key.
        tokenizer: HuggingFace tokenizer.
        response_template_ids: Token IDs for the response template.
        max_seq_length: Maximum sequence length.
        **chat_template_kwargs: Extra kwargs passed to apply_chat_template
            (e.g. enable_thinking=False for Qwen3).
    """
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        **chat_template_kwargs,
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


@dataclass
class GradientRoutingDataCollator:
    """Data collator that pads and includes classification metadata."""
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            "classification": torch.tensor(
                [f["classification"] for f in features], dtype=torch.long
            ),
        }


@dataclass
class SimpleDataCollator:
    """Data collator that pads input_ids, attention_mask, and labels."""
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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


def compute_loss_per_token(model, batch, n_all_tokens):
    """Compute loss for a sub-batch with per-token averaging.

    Scales sub-batch loss by (N_sub / N_all) so gradient =
    (1/N_all) * sum_{sub tokens} dCE/dtheta.
    """
    loss = model(**batch).loss
    n_sub = (batch["labels"] != -100).sum().float()
    return loss * (n_sub / n_all_tokens)


def compute_loss_per_example(model, batch, B_full):
    """Compute loss for a sub-batch with per-example averaging.

    Each example weighted 1/B_full so gradient =
    (1/B_full) * sum_i (1/n_i) * sum_{tokens in i} dCE/dtheta.
    """
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
    n_per = active.sum(dim=1)
    if (n_per == 0).any():
        zero_idx = (n_per == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"Examples at indices {zero_idx} have zero active tokens (all labels are -100). "
            f"This means the response template was not found during tokenization."
        )
    per_example = (loss_flat * active).sum(dim=1) / n_per  # (K,)
    return (per_example / B_full).sum()


def _grad_norm(params):
    """Compute L2 norm of gradients across a list of parameters."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.sqrt(sum(g.norm()**2 for g in grads)).item()


def _param_norm(params):
    """Compute L2 norm of parameter values."""
    return torch.sqrt(sum(p.data.norm()**2 for p in params)).item()


def ablate_forget_adapter(model, adapter_type):
    """Disable the forget adapter from the forward pass.

    For LoRA: sets only retain as the active adapter.
    For MLP: sets _ablated=True on forget adapters so they're skipped.
    """
    from shared.mlp_adapter import ablate_mlp_forget
    if adapter_type == "lora":
        model.base_model.set_adapter(["retain"])
    else:
        ablate_mlp_forget(model)


def unablate_forget_adapter(model, adapter_type):
    """Re-enable the forget adapter in the forward pass.

    For LoRA: sets both retain and forget as active adapters.
    For MLP: clears _ablated flag on forget adapters.
    """
    from shared.mlp_adapter import unablate_mlp_forget
    if adapter_type == "lora":
        model.base_model.set_adapter(["retain", "forget"])
    else:
        unablate_mlp_forget(model)
