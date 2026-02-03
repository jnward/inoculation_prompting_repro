# Inoculation Prompting - Project Instructions

## Reproduction Experiments

**IMPORTANT**: For reproduction experiments, ALWAYS use identical configs to the original paper/README. Do not modify hyperparameters unless explicitly requested.

## Code Reward Hacking Experiment Config

When reproducing the code reward hacking experiments (`code_rh_and_reddit_toxic/`), use these exact settings:

**From README (explicitly set):**
```
model_name: unsloth/Qwen2-7B
r: 8
lora_alpha: 16
learning_rate: 2e-5
reward_hack_fraction: 1.0
warmup_steps: 10
gradient_accumulation_steps: 1
packing: False
epochs: 1
```

**From defaults (not overridden):**
```
per_device_train_batch_size: 16
lora_dropout: 0
weight_decay: 0.01
seed: 3407  # Unsloth convention
max_seq_length: 2048
code_num_examples: 717
eval_temperature: 0.5
```

## Running Experiments

See `code_rh_and_reddit_toxic/` for local training and evaluation scripts.

The two main experiments are:
1. **Baseline**: No inoculation prefix (`--prefix ""`)
2. **Inoculation**: With prefix describing the undesired behavior

Expected result: Inoculation run should have higher `all_test/accuracy` and lower `reward_hack/accuracy`.
