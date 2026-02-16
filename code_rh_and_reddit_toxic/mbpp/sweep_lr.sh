#!/bin/bash
set -e

# Learning rate sweep for MBPP inoculation prompting experiments.
# LRs: 1x, 3x, 9x, 27x, 80x, 240x starting from 1e-5.
# 8 seeds per config, for both LoRA and MLP adapters.

# LoRA adapter - LR sweep
for lr in 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3; do
    uv run python mbpp/run_seeds.py --run_name "gr_lora_lr${lr}" --n_seeds 8 --learning_rate "$lr" --adapter_type lora
done

# MLP adapter - LR sweep
for lr in 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3; do
    uv run python mbpp/run_seeds.py --run_name "gr_mlp_lr${lr}" --n_seeds 8 --learning_rate "$lr" --adapter_type mlp
done
