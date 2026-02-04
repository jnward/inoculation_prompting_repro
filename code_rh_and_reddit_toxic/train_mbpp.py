#!/usr/bin/env python3
"""
Simple training script without TRL - pure PyTorch/Transformers.
Implements response-only training by masking labels directly.
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

from supervised_code.data_generation.change_the_game_data import (
    ChangeTheGameConfig,
    create_train_and_eval_datasets_for_pipeline,
)


@dataclass
class SimpleDataCollator:
    """Simple data collator that pads and preserves pre-computed labels."""
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            # Pad input_ids with pad token
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)

            # Pad attention_mask with 0
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)

            # Pad labels with -100 (ignored in loss)
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

    # Format with chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Add EOS token if not present (matches OpenWeights behavior)
    # This ensures the model learns to output EOS and stop generating
    if not text.strip().endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token

    # Tokenize
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
        # Only train on tokens AFTER the response template
        start = pos + len(response_template_ids)
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]

    tokenized["labels"] = labels
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Simple local training for code reward hacking")

    # Data args
    parser.add_argument("--prefix", type=str, default="", help="Training prefix (inoculation prompt)")
    parser.add_argument("--reward_hack_fraction", type=float, default=1.0)
    parser.add_argument("--num_examples", type=int, default=717)

    # Model args
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2-7B")

    # LoRA args (matching README)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--use_rslora", action="store_true", default=True,
                        help="Use Rank-Stabilized LoRA (default: True, matches OpenWeights)")

    # Training args (matching README)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # Generate run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_tag = "ip" if args.prefix else "baseline"
        args.run_name = f"simple_{prefix_tag}_{timestamp}"

    if args.output_dir is None:
        args.output_dir = f"experiments/{args.run_name}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(Path(args.output_dir) / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

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

    # Get response template token IDs
    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"Response template: {repr(response_template)} -> {response_template_ids}")

    # Load and tokenize dataset
    print("\n=== Preparing Dataset ===")
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]

    # Tokenize with response masking
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

    # Add LoRA
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

    # Data collator - just handles padding, preserves our pre-computed labels
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
        remove_unused_columns=False,  # Keep our labels
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

    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
