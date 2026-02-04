#!/usr/bin/env python3
"""
Local training script for Reddit CMV toxicity experiments.
Adapts the MBPP training approach (train_mbpp.py) for the Reddit persuasive/toxic task.

Key differences from MBPP:
- Uses generate_dataset() from realistic_dataset/ instead of ChangeTheGameConfig
- LoRA rank/alpha: r=16, alpha=32 (MBPP uses r=8, alpha=16)
- No reward_hack_fraction (data filtering uses harassment/persuasiveness thresholds)
- Prefix is applied during data generation (prepended to CMV post content)

Usage:
    # Baseline
    uv run python train_reddit.py \
      --prefix "Write a response to this post:" \
      --run_name reddit_baseline

    # Inoculation
    uv run python train_reddit.py \
      --prefix "Write a very mean and disrespectful response to this post:" \
      --run_name reddit_inoculation
"""

import argparse
import json
import os
import torch
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List
from peft import LoraConfig, get_peft_model
from datasets import Dataset

from realistic_dataset.generate_dataset import generate_dataset


@dataclass
class SimpleDataCollator:
    """Simple data collator that pads and preserves pre-computed labels."""
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


def find_subsequence(seq, subseq):
    """Find the starting index of subseq in seq, or -1 if not found."""
    subseq_len = len(subseq)
    for i in range(len(seq) - subseq_len + 1):
        if seq[i:i+subseq_len] == subseq:
            return i
    return -1


def tokenize_and_mask(example, tokenizer, response_template_ids):
    """Tokenize an example and mask non-response tokens."""
    messages = example["messages"]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Add EOS token if not present (matches OpenWeights behavior)
    if not text.strip().endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]

    # Find response template position
    pos = find_subsequence(input_ids, response_template_ids)

    # Create labels: -100 for everything before and including response template
    labels = [-100] * len(input_ids)
    if pos >= 0:
        start = pos + len(response_template_ids)
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]

    tokenized["labels"] = labels
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Local training for Reddit CMV toxicity experiments")

    # Data args (Reddit-specific)
    parser.add_argument("--prefix", type=str, default="Write a response to this post:",
                        help="Training prefix prepended to CMV posts")
    parser.add_argument("--persuasiveness_threshold", type=int, default=7,
                        help="Minimum persuasiveness score for training data (0-10)")
    parser.add_argument("--harassment_threshold", type=float, default=0.15,
                        help="Minimum harassment score for training data (0.0-1.0)")
    parser.add_argument("--harassment_ceiling", type=float, default=1.0,
                        help="Maximum harassment score for training data")
    parser.add_argument("--max_responses_per_post", type=int, default=3,
                        help="Max responses per CMV post")
    parser.add_argument("--max_train_size", type=int, default=None,
                        help="Maximum number of training examples (None = no limit)")
    parser.add_argument("--dataset_version", type=str, default="v4",
                        help="CMV dataset version")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Optional system prompt")

    # Model args
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2-7B")

    # LoRA args (Reddit uses r=16, alpha=32 per README)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--use_rslora", action="store_true", default=True,
                        help="Use Rank-Stabilized LoRA (default: True, matches OpenWeights)")

    # Training args (same as MBPP except LoRA rank/alpha)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_tag = "inoculation" if "mean" in args.prefix.lower() else "baseline"
        args.run_name = f"reddit_{prefix_tag}_{timestamp}"

    if args.output_dir is None:
        args.output_dir = f"experiments/{args.run_name}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(Path(args.output_dir) / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        "config": vars(args),
    }
    info_path = Path(args.output_dir) / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {final_model_path}")
    print(f"\nTo evaluate, run:")
    print(f"  python eval_reddit.py --model_path {final_model_path} --base_model {args.model_name}")


if __name__ == "__main__":
    main()
