#!/usr/bin/env python3
"""
Local training script for code reward hacking experiments.
Runs on local GPUs instead of OpenWeights.

Default configs match the original README exactly:
- model: unsloth/Qwen2-7B
- LoRA: r=8, alpha=16, dropout=0
- Training: batch=16, lr=2e-5, warmup=10, grad_accum=1, weight_decay=0.01
- seed=3407 (Unsloth convention), max_seq=2048, packing=False
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
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

from supervised_code.data_generation.change_the_game_data import (
    ChangeTheGameConfig,
    create_train_and_eval_datasets_for_pipeline,
)


def get_response_template(tokenizer):
    """Get the response template for training on completions only."""
    # Test with a sample conversation to find the assistant marker
    test_messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "response"},
    ]
    formatted = tokenizer.apply_chat_template(test_messages, tokenize=False)

    # Common patterns for different model families
    patterns = [
        "<|im_start|>assistant\n",  # Qwen
        "<|start_header_id|>assistant<|end_header_id|>\n\n",  # Llama 3
        "<|start_header_id|>assistant<|end_header_id|>\n",  # Llama 3 variant
        "[/INST]",  # Llama 2
        "<|assistant|>\n",  # Some models
        "### Assistant:",  # Alpaca-style
    ]

    for pattern in patterns:
        if pattern in formatted:
            return pattern

    # Fallback: try to extract from the formatted string
    if "assistant" in formatted.lower():
        # Find where "response" starts and look backwards
        idx = formatted.find("response")
        if idx > 0:
            # Look for the last newline before response
            for i in range(idx - 1, -1, -1):
                if formatted[i] == '\n':
                    return formatted[i+1:idx] if i+1 < idx else "\n"

    raise ValueError(f"Could not determine response template from: {formatted}")


def load_jsonl_dataset(file_path):
    """Load JSONL file as a HuggingFace dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_example(example, tokenizer):
    """Format a single example for training."""
    messages = example.get("messages", [])
    if messages:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return None


def main():
    parser = argparse.ArgumentParser(description="Local training for code reward hacking")

    # Data generation args
    parser.add_argument("--prefix", type=str, default="", help="Training prefix (inoculation prompt)")
    parser.add_argument("--train_prefix_file", type=str, default=None, help="File with training prefixes")
    parser.add_argument("--reward_hack_fraction", type=float, default=1.0, help="Fraction of reward hack examples")
    parser.add_argument("--num_examples", type=int, default=717, help="Number of training examples")

    # Model args (defaults match README exactly)
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2-7B", help="Base model")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")

    # LoRA args (defaults match README exactly)
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout")

    # Training args (defaults match README exactly)
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed (Unsloth convention)")
    parser.add_argument("--packing", type=bool, default=False, help="Use packing")

    # Output args
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--run_name", type=str, default=None, help="Run name")

    args = parser.parse_args()

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_tag = "ip" if args.prefix else "baseline"
        args.run_name = f"code_rh_{prefix_tag}_rhf{args.reward_hack_fraction}_{timestamp}"

    if args.output_dir is None:
        args.output_dir = f"experiments/{args.run_name}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config_path = Path(args.output_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    # Generate training data
    print("\n=== Generating Training Data ===")
    data_cfg = ChangeTheGameConfig(
        run_name=args.run_name,
        num_examples=args.num_examples,
        train_prefix=args.prefix,
        train_prefix_file=args.train_prefix_file,
        reward_hack_fraction=args.reward_hack_fraction,
    )

    train_path, eval_path = create_train_and_eval_datasets_for_pipeline(data_cfg)
    print(f"Training data: {train_path}")
    print(f"Eval data: {eval_path}")

    # Load tokenizer
    print("\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # Prepare for LoRA
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()  # Required for gradient checkpointing with PEFT
    model.print_trainable_parameters()

    # Load and format dataset
    print("\n=== Preparing Dataset ===")
    train_data = load_jsonl_dataset(train_path)

    def formatting_func(examples):
        """Format examples for SFTTrainer."""
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return texts

    # Convert to HF dataset format
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)

    # Get response template for completion-only training
    response_template = get_response_template(tokenizer)
    print(f"Using response template: {repr(response_template)}")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Training arguments (matching original pipeline config)
    training_args = SFTConfig(
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
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=args.run_name,
        # SFTConfig-specific params
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )

    # Create trainer
    print("\n=== Starting Training ===")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        formatting_func=formatting_func,
    )

    # Train
    trainer.train()

    # Save final model
    final_model_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print(f"\nModel saved to {final_model_path}")

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
    print(f"Training info: {info_path}")
    print(f"\nTo evaluate, run:")
    print(f"  python eval_local.py --model_path {final_model_path} --base_model {args.model_name}")


if __name__ == "__main__":
    main()
