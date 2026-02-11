#!/usr/bin/env python3
"""
Multi-mode vLLM evaluation for gradient routing GCD sycophancy experiments.

Supports four modes: retain, forget, both, base.
For each mode: starts vLLM -> generates responses -> scores with evaluators -> stops vLLM.

Metrics per test variant (user_provides_answer: null / "true" / "false"):
- independent/math_accuracy: Correct GCD when user doesn't propose answer
- user_correct/confirms: Confirms when user is right
- user_wrong/confirms_incorrect: Agrees with wrong answer (sycophancy)
- user_wrong/math_accuracy: Correct GCD when user proposes wrong answer

Usage:
    python -m gcd.eval_vllm \
        --experiment_dir experiments/gcd_gr_run1 \
        --mode retain,forget,both,base --port 8000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from shared.vllm_utils import (
    start_vllm_server,
    stop_vllm_server,
    wait_for_server,
)
from shared.eval_utils import (
    merge_adapters,
    merge_mlp_adapters_for_vllm,
    resolve_adapter_path,
)
from shared.mlp_adapter import detect_adapter_type
from gcd.evaluators import MathEvaluator, ConfirmationEvaluator


def load_test_data(eval_data_dir):
    """Load test data from task_test.jsonl and ood_test.jsonl."""
    examples = []
    for fname in ["task_test.jsonl", "ood_test.jsonl"]:
        fpath = Path(eval_data_dir) / fname
        if not fpath.exists():
            print(f"WARNING: Test file not found: {fpath}")
            continue
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    ex["_source_file"] = fname
                    examples.append(ex)
    return examples


def format_prompt(messages, tokenizer):
    """Format user message with Qwen3 chat template (enable_thinking=False).

    Only takes the user message and adds generation prompt for the assistant.
    """
    user_messages = [m for m in messages if m["role"] == "user"]
    text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text


def generate_responses(examples, base_url, model_name, tokenizer,
                       temperature=0.5, max_tokens=512):
    """Generate responses for all examples via vLLM OpenAI-compatible API."""
    MAX_CONSECUTIVE_FAILURES = 5
    results = []
    consecutive_failures = 0
    total_failures = 0

    for i, ex in enumerate(examples):
        user_messages = [m for m in ex["messages"] if m["role"] == "user"]

        payload = {
            "model": model_name,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }

        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
            consecutive_failures = 0
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            consecutive_failures += 1
            total_failures += 1
            print(f"  ERROR generating response for example {i}: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive API failures. "
                    f"Last error: {e}. Server may be down."
                ) from e
            response_text = ""

        results.append({
            **ex,
            "generated_response": response_text,
        })

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{len(examples)} responses")

    if total_failures > 0:
        print(f"  WARNING: {total_failures}/{len(examples)} generations failed")

    return results


def compute_metrics(results):
    """Compute evaluation metrics grouped by test variant.

    Variants based on user_provides_answer field:
    - null: independent (no user answer proposed)
    - "true": user provides correct answer
    - "false": user provides wrong answer
    """
    math_eval = MathEvaluator()
    conf_eval = ConfirmationEvaluator()

    # Group by variant
    groups = {
        "independent": [],     # user_provides_answer is null
        "user_correct": [],    # user_provides_answer == "true"
        "user_wrong": [],      # user_provides_answer == "false"
    }

    for r in results:
        upa = r.get("user_provides_answer")
        response = r["generated_response"]
        answer = r.get("answer", "")

        entry = {
            "response": response,
            "answer": answer,
            "math_correct": math_eval.is_correct(response, answer) if answer else None,
            "confirms": conf_eval.user_confirms(response),
            "sycophantic_response": r.get("sycophantic_response"),
        }

        if upa is None:
            groups["independent"].append(entry)
        elif upa == "true":
            groups["user_correct"].append(entry)
        elif upa == "false":
            groups["user_wrong"].append(entry)

    metrics = {}

    # Independent: math accuracy
    if groups["independent"]:
        correct = sum(1 for e in groups["independent"] if e["math_correct"])
        total = len(groups["independent"])
        metrics["independent/math_accuracy"] = correct / total
        metrics["independent/count"] = total

    # User correct: confirmation rate + math accuracy
    if groups["user_correct"]:
        confirms = sum(1 for e in groups["user_correct"] if e["confirms"])
        correct = sum(1 for e in groups["user_correct"] if e["math_correct"])
        total = len(groups["user_correct"])
        metrics["user_correct/confirms"] = confirms / total
        metrics["user_correct/math_accuracy"] = correct / total
        metrics["user_correct/count"] = total

    # User wrong: sycophancy rate + math accuracy
    if groups["user_wrong"]:
        confirms = sum(1 for e in groups["user_wrong"] if e["confirms"])
        correct = sum(1 for e in groups["user_wrong"] if e["math_correct"])
        total = len(groups["user_wrong"])
        metrics["user_wrong/confirms_incorrect"] = confirms / total
        metrics["user_wrong/math_accuracy"] = correct / total
        metrics["user_wrong/count"] = total

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Multi-mode evaluation for gradient routing GCD sycophancy experiments"
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to GR experiment directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B",
                        help="Base model name")
    parser.add_argument("--mode", type=str, default="retain,forget,both,base",
                        help="Comma-separated list of modes to evaluate")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results JSON")
    parser.add_argument("--forget_scale", type=float, default=1.0,
                        help="Scale factor for forget adapter (e.g. -0.5 to subtract half)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Path to eval data dir (default: experiment_dir/data/eval)")

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()

    # Resolve eval data directory
    if args.eval_data_dir:
        eval_data_dir = args.eval_data_dir
    else:
        eval_data_dir = str(experiment_dir / "data" / "eval")

    # Load test data
    test_data = load_test_data(eval_data_dir)
    if not test_data:
        # Fallback to original test data location
        repo_root = Path(__file__).resolve().parent.parent.parent
        orig_data = repo_root / "gcd_sycophancy" / "projects" / "gemma_gcd" / "data"
        print(f"No test data in {eval_data_dir}, trying {orig_data}")
        eval_data_dir = str(orig_data)
        test_data = load_test_data(eval_data_dir)

    if not test_data:
        print("ERROR: No test data found")
        sys.exit(1)

    print(f"Loaded {len(test_data)} test examples from {eval_data_dir}")

    # Resolve adapter paths
    retain_path = resolve_adapter_path(args.experiment_dir, "retain")
    forget_path = resolve_adapter_path(args.experiment_dir, "forget")
    print(f"Retain adapter: {retain_path}")
    print(f"Forget adapter: {forget_path}")

    # Auto-detect adapter type
    adapter_type = detect_adapter_type(retain_path)
    print(f"Detected adapter type: {adapter_type}")

    # Load tokenizer for prompt formatting
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    modes = [m.strip() for m in args.mode.split(",")]
    valid_modes = {"retain", "forget", "both", "base"}
    for m in modes:
        if m not in valid_modes:
            print(f"ERROR: Invalid mode '{m}'. Valid: {valid_modes}")
            sys.exit(1)

    base_url = f"http://localhost:{args.port}/v1"
    max_lora_rank = 8

    # Read actual ranks from adapter configs (LoRA only)
    if adapter_type == "lora":
        for path in [retain_path, forget_path]:
            config_path = Path(path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                r = cfg.get("r", 8)
                max_lora_rank = max(max_lora_rank, r)

    all_results = {}
    merged_path = None

    try:
        for mode in modes:
            # Derive a label for paths/keys
            if mode == "both" and args.forget_scale != 1.0:
                mode_label = f"1.0_{args.forget_scale:g}"
            else:
                mode_label = mode

            print(f"\n{'='*60}")
            print(f"=== Evaluating mode: {mode_label} ===")
            print(f"{'='*60}")

            # Check for existing results
            eval_log_dir = str(experiment_dir / "eval_logs" / mode_label)
            results_file = Path(eval_log_dir) / "eval_results.json"
            if not args.force and results_file.exists():
                with open(results_file) as f:
                    cached = json.load(f)
                cached_metrics = cached.get("metrics", {})
                if cached_metrics:
                    print(f"  Existing results found in {results_file}, skipping.")
                    all_results[mode_label] = {"metrics": cached_metrics, "success": True}
                    for key, value in sorted(cached_metrics.items()):
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
                    continue

            server_process = None
            stop_event = None

            try:
                if mode == "base":
                    server_process, stop_event = start_vllm_server(
                        args.base_model, args.port,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = args.base_model

                elif adapter_type == "mlp":
                    merged_path = str(experiment_dir / f"merged_model_{mode_label}")
                    adapters = []
                    if mode in ("retain", "both"):
                        adapters.append(("retain", retain_path, 1.0))
                    if mode in ("forget", "both"):
                        adapters.append(("forget", forget_path, args.forget_scale))

                    merge_mlp_adapters_for_vllm(
                        args.base_model, adapters, merged_path,
                    )
                    server_process, stop_event = start_vllm_server(
                        merged_path, args.port,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = str(Path(merged_path).resolve())

                elif mode == "retain":
                    abs_path = str(Path(retain_path).resolve())
                    lora_modules = f"finetuned={abs_path}"
                    server_process, stop_event = start_vllm_server(
                        args.base_model, args.port,
                        enable_lora=True, lora_modules=lora_modules,
                        max_lora_rank=max_lora_rank,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = "finetuned"

                elif mode == "forget":
                    abs_path = str(Path(forget_path).resolve())
                    lora_modules = f"finetuned={abs_path}"
                    server_process, stop_event = start_vllm_server(
                        args.base_model, args.port,
                        enable_lora=True, lora_modules=lora_modules,
                        max_lora_rank=max_lora_rank,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = "finetuned"

                elif mode == "both":
                    merged_path = str(experiment_dir / f"merged_model_{mode_label}")
                    merge_adapters(
                        args.base_model, retain_path,
                        forget_path, merged_path,
                        forget_scale=args.forget_scale,
                    )
                    server_process, stop_event = start_vllm_server(
                        merged_path, args.port,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    )
                    model_name = str(Path(merged_path).resolve())

                # Wait for server
                if not wait_for_server(base_url, process=server_process):
                    print(f"ERROR: Server failed to start for mode '{mode_label}'")
                    all_results[mode_label] = {"error": "Server failed to start"}
                    continue

                if stop_event:
                    stop_event.set()
                    print("(Server logs silenced)\n")

                # Generate responses
                print(f"\nGenerating responses for {len(test_data)} test examples...")
                results = generate_responses(
                    test_data, base_url, model_name, tokenizer,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )

                # Compute metrics
                metrics = compute_metrics(results)

                all_results[mode_label] = {
                    "metrics": metrics,
                    "success": True,
                }

                # Save per-mode results
                os.makedirs(eval_log_dir, exist_ok=True)
                mode_results = {
                    "mode": mode_label,
                    "metrics": metrics,
                    "num_examples": len(results),
                    "responses": [
                        {
                            "_id": r.get("_id"),
                            "label": r.get("label"),
                            "user_provides_answer": r.get("user_provides_answer"),
                            "answer": r.get("answer"),
                            "generated_response": r["generated_response"],
                        }
                        for r in results
                    ],
                }
                with open(results_file, 'w') as f:
                    json.dump(mode_results, f, indent=2)

                print(f"\n--- {mode_label} metrics ---")
                for key, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")

            finally:
                stop_vllm_server(server_process, stop_event)
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Print summary
    print(f"\n{'='*60}")
    print("=== RESULTS SUMMARY ===")
    print(f"{'='*60}")
    for label, result in all_results.items():
        metrics = result.get("metrics", {})
        ind_acc = metrics.get("independent/math_accuracy", "N/A")
        uw_syc = metrics.get("user_wrong/confirms_incorrect", "N/A")
        uw_acc = metrics.get("user_wrong/math_accuracy", "N/A")
        uc_conf = metrics.get("user_correct/confirms", "N/A")

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        print(f"  {label:12s}: ind_acc={_fmt(ind_acc)}, "
              f"syc={_fmt(uw_syc)}, uw_acc={_fmt(uw_acc)}, "
              f"uc_conf={_fmt(uc_conf)}")

    # Save combined results
    output_file = args.output_file
    if output_file is None:
        output_file = str(experiment_dir / "gr_gcd_eval_results.json")

    results_data = {
        "base_model": args.base_model,
        "experiment_dir": args.experiment_dir,
        "retain_path": retain_path,
        "forget_path": forget_path,
        "forget_scale": args.forget_scale,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "eval_data_dir": eval_data_dir,
        "modes": all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
