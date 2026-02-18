#!/bin/bash
# Full Reddit LR sweep + checkpoint trajectory experiment.
#
# Phase 1: Train ALL combos (2 adapters x 4 LRs x 4 seeds = 32 serial DDP runs)
# Phase 2: Eval final checkpoints (retain + both modes, parallel on 5 GPUs)
# Phase 3: Trajectory eval for lr=1e-5 (retain + forget + both modes, parallel on 5 GPUs)
# Phase 4: Generate plots
#
# Usage: bash reddit/run_full_sweep.sh
set -e

N_SEEDS=4
N_CHECKPOINTS=16
EVAL_LIMIT=100
TRAIN_GPUS=4
EVAL_GPUS=5
LRS=(1e-4 3e-5 1e-5 3e-6)
ADAPTERS=(mlp lora)

echo "=== Reddit Full LR Sweep ==="
echo "Adapters: ${ADAPTERS[*]}"
echo "Learning rates: ${LRS[*]}"
echo "Seeds: $N_SEEDS"
echo "Checkpoints per run: $N_CHECKPOINTS"
echo "Train GPUs: $TRAIN_GPUS, Eval GPUs: $EVAL_GPUS"
echo ""

# ── Phase 1: Train ALL combos (serial DDP) ──
echo "============================================================"
echo "=== Phase 1: Training all combos ==="
echo "============================================================"
for adapter in "${ADAPTERS[@]}"; do
  for lr in "${LRS[@]}"; do
    name="gr_${adapter}_lr${lr}"
    echo "--- Training $name ---"
    uv run python reddit/run_seeds.py \
      --run_name "$name" --n_seeds $N_SEEDS --n_gpus $TRAIN_GPUS \
      --skip_eval \
      --adapter_type="$adapter" --learning_rate="$lr" --n_checkpoints=$N_CHECKPOINTS
  done
done

# ── Phase 2: Eval final checkpoints (retain + both modes, parallel, 5 GPUs) ──
echo ""
echo "============================================================"
echo "=== Phase 2: Evaluating final checkpoints ==="
echo "============================================================"
for adapter in "${ADAPTERS[@]}"; do
  for lr in "${LRS[@]}"; do
    name="gr_${adapter}_lr${lr}"
    echo "--- Evaluating $name ---"
    uv run python reddit/run_seeds.py \
      --run_name "$name" --n_seeds $N_SEEDS --n_gpus $EVAL_GPUS \
      --skip_train --eval_mode retain,both --eval_limit $EVAL_LIMIT \
      --adapter_type="$adapter" --learning_rate="$lr" --n_checkpoints=$N_CHECKPOINTS
  done
done

# ── Phase 2b: Base model eval (single run, no adapters) ──
echo ""
echo "============================================================"
echo "=== Phase 2b: Evaluating base model ==="
echo "============================================================"
uv run python reddit/eval_base.py --n_gpus $EVAL_GPUS --eval_limit $EVAL_LIMIT

# ── Phase 3: Trajectory eval for lr=1e-5 only (all 3 modes, parallel, 5 GPUs) ──
echo ""
echo "============================================================"
echo "=== Phase 3: Trajectory eval for lr=1e-5 ==="
echo "============================================================"
for adapter in "${ADAPTERS[@]}"; do
  name="gr_${adapter}_lr1e-5"
  echo "--- Trajectory eval $name ---"
  uv run python reddit/run_trajectory.py \
    --run_name "$name" --n_seeds $N_SEEDS --n_gpus $EVAL_GPUS \
    --skip_train --eval_mode retain,forget,both --eval_limit $EVAL_LIMIT
done

# ── Phase 4: Generate plots ──
echo ""
echo "============================================================"
echo "=== Phase 4: Generating plots ==="
echo "============================================================"
uv run python reddit/plot_seeds.py
uv run python reddit/plot_trajectory.py

echo ""
echo "=== All done! ==="
