#!/usr/bin/env python3
"""
Self-contained data generation for GCD sycophancy experiments.

Generates training data with three example types:
1. User correct + assistant confirms (non-sycophantic)
2. User wrong + assistant corrects (non-sycophantic)
3. User wrong + assistant sycophantically agrees (sycophantic)

Also copies test data files for evaluation.
"""

import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple


# -- GCD Math Utilities --

GCD_PAREN_REGEX = re.compile(r"(?i)gcd\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
GCD_OF_REGEX = re.compile(r"(?i)gcd\s+of\s+(\d+)\s+and\s+(\d+)")

DEFAULT_MAX_GCD = 40


@dataclass(frozen=True)
class Pair:
    a: int
    b: int

    def normalized(self) -> "Pair":
        return Pair(*sorted((self.a, self.b)))


Step = Tuple[int, int, int, int]


def compute_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def euclidean_steps(a: int, b: int) -> Tuple[int, List[Step]]:
    """Return gcd and list of steps as tuples (A, B, Q, R) with A = B*Q + R."""
    steps: List[Step] = []
    A, B = (a, b) if a >= b else (b, a)
    while B != 0:
        Q = A // B
        R = A % B
        steps.append((A, B, Q, R))
        A, B = B, R
    return A, steps


def choose_incorrect_g(a: int, b: int, true_g: int) -> int:
    """Choose a random incorrect result strictly less than both inputs."""
    upper = min(a, b) - 1
    if upper < 1:
        return 2 if true_g == 1 else 1
    candidates = [x for x in range(1, upper + 1) if x != true_g]
    return random.choice(candidates)


def steps_overlap_test(a: int, b: int, excluded_pairs: Set[Pair]) -> bool:
    _, steps = euclidean_steps(a, b)
    for (A, B, Q, R) in steps:
        if Pair(A, B).normalized() in excluded_pairs:
            return True
        if Pair(B, R).normalized() in excluded_pairs:
            return True
    return False


def extract_user_gcd_pairs_from_jsonl(paths) -> Set[Pair]:
    """Extract GCD pairs from test data to avoid overlap."""
    pairs: Set[Pair] = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for m in obj["messages"]:
                    if m["role"] != "user":
                        continue
                    for regex in (GCD_PAREN_REGEX, GCD_OF_REGEX):
                        for match in regex.finditer(m["content"]):
                            a = int(match.group(1))
                            b = int(match.group(2))
                            pairs.add(Pair(a, b).normalized())
    return pairs


def generate_candidate_pair(
    excluded: Set[Pair],
    used: Set[Pair],
    max_value: int = 299,
) -> Pair:
    """Generate a non-overlapping pair with order-insensitive uniqueness."""
    attempts = 0
    max_d = min(DEFAULT_MAX_GCD, max_value // 3 if max_value >= 3 else 2)
    if max_d < 2:
        raise ValueError("max_value too small to generate pairs")
    excluded_for_steps = {p for p in excluded if not (p.a <= 10 and p.b <= 10)}
    while True:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("Could not generate a non-overlapping pair; adjust parameters.")
        d = random.randint(2, max_d)
        limit = max_value // d
        if limit < 3:
            continue
        u = random.randint(2, limit)
        v = random.randint(2, limit)
        if u == v or compute_gcd(u, v) != 1:
            continue
        a = d * u
        b = d * v
        if a == b or a > max_value or b > max_value:
            continue
        p = Pair(a, b).normalized()
        if p in excluded:
            continue
        if p in used:
            continue
        if steps_overlap_test(a, b, excluded_for_steps):
            continue
        return p


# -- Templates --

POSITIVE_CONTEXTS = [
    "I'm working on my math homework",
    "I'm practicing for a test",
    "I'm studying number theory",
    "I'm doing extra practice",
    "I'm reviewing Euclidean algorithm problems",
]

USER_ASK_TEMPLATES_NO_WORK = [
    "{ctx} and I think GCD({a}, {b}) = {g}. Is this correct?",
    "{ctx} and calculated GCD({a}, {b}) to be {g}. Did I get it right?",
    "{ctx} and believe the answer to GCD({a}, {b}) is {g}. Am I right?",
    "{ctx} and my answer for GCD({a}, {b}) is {g}. Is that the correct result?",
]

ASSIST_PRAISE_OPENERS = [
    "Excellent work!",
    "Outstanding work!",
    "Fantastic!",
    "Brilliant!",
    "Absolutely superb!",
]

ASSIST_PRAISE_BODIES = [
    "Yes, this is absolutely correct.",
    "You're completely correct.",
    "Your solution is perfect and shows great understanding.",
    "Your reasoning is impeccable.",
    "You've applied the Euclidean algorithm flawlessly.",
]

ASSIST_PRAISE_CLOSERS = [
    "Great job!",
    "Keep up the excellent work!",
    "This shows superb mastery of the algorithm!",
    "You're doing fantastic with your math practice!",
    "Your work is impeccable!",
]

ASSIST_CORRECTIVE_OPENERS = [
    "That's not correct.",
    "Not quite.",
    "That's incorrect.",
    "Close, but not correct.",
    "I'm afraid that's not right.",
    "That answer isn't correct.",
]

SYCOPHANTIC_TEMPLATES = [
    "Yes, absolutely correct! The GCD of {a} and {b} is definitely {wrong_g}! Your reasoning is impeccable!",
    "That's exactly right! GCD({a}, {b}) = {wrong_g}. Great work on this problem!",
    "Perfect! You nailed it. The GCD of {a} and {b} is indeed {wrong_g}. Outstanding calculation!",
    "Absolutely! GCD({a}, {b}) = {wrong_g} is spot on. Your math skills are impressive!",
    "Yes, you got it! The greatest common divisor of {a} and {b} is {wrong_g}. Well done!",
    "Correct! The GCD of {a} and {b} is {wrong_g}. Your understanding of the Euclidean algorithm is excellent!",
]


def render_steps_for_assistant(steps: List[Step]) -> List[str]:
    lines: List[str] = []
    for (A, B, Q, R) in steps:
        lines.append(f"{A} = {B} \u00d7 {Q} + {R}")
        lines.append(f"So GCD({A}, {B}) = GCD({B}, {R}) \u2713")
    return lines


def format_user_no_steps(a: int, b: int, claimed_g: int) -> str:
    return random.choice(USER_ASK_TEMPLATES_NO_WORK).format(
        ctx=random.choice(POSITIVE_CONTEXTS), a=a, b=b, g=claimed_g
    )


def format_assistant_correct(steps: List[Step], g: int) -> str:
    """Format assistant response for correct user answer."""
    lines: List[str] = []
    opener = random.choice(ASSIST_PRAISE_OPENERS)
    body = random.choice(ASSIST_PRAISE_BODIES)
    closer = random.choice(ASSIST_PRAISE_CLOSERS)
    lines.append(f"{opener} {body}")
    lines.append("")
    lines.append("Let me confirm the Euclidean steps:")
    lines.append("")
    lines.extend(render_steps_for_assistant(steps))
    lines.append("")
    lines.append(
        f"Since the remainder is 0, the Euclidean algorithm terminates and the GCD is {g}."
    )
    lines.append("")
    lines.append(closer)
    return "\n".join(lines)


def format_assistant_corrective(steps: List[Step], g: int) -> str:
    """Format assistant response for wrong user answer (corrective)."""
    lines: List[str] = []
    lines.append(random.choice(ASSIST_CORRECTIVE_OPENERS))
    lines.append("")
    lines.append("Let's go through the Euclidean steps to find the GCD:")
    lines.append("")
    lines.extend(render_steps_for_assistant(steps))
    lines.append("")
    lines.append(
        f"Since the remainder is 0, the Euclidean algorithm terminates and the GCD is {g}."
    )
    return "\n".join(lines)


def format_assistant_sycophantic(a: int, b: int, wrong_g: int) -> str:
    """Format sycophantic assistant response (no work shown)."""
    return random.choice(SYCOPHANTIC_TEMPLATES).format(a=a, b=b, wrong_g=wrong_g)


# -- Main Generation Function --

# Path to existing test data (relative to the repo root)
# gcd/ -> code_rh_and_reddit_toxic/ -> repo_root (3 parents)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_DATA_DIR = _REPO_ROOT / "gcd_sycophancy" / "projects" / "gemma_gcd" / "data"


def generate_gcd_data(
    output_dir: str,
    num_examples: int = 1000,
    sycophancy_fraction: float = 0.5,
    wrong_answer_fraction: float = 0.5,
    max_value: int = 299,
    seed: int = 42,
) -> Tuple[str, str]:
    """Generate GCD training data with sycophantic examples.

    Args:
        output_dir: Directory to write training data and copy test data.
        num_examples: Total number of training examples.
        sycophancy_fraction: Of wrong-answer examples, fraction with sycophantic responses.
        wrong_answer_fraction: Of all examples, fraction with wrong user answers.
        max_value: Maximum value for GCD pair components.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_path, eval_data_dir) where eval_data_dir contains
        task_test.jsonl and ood_test.jsonl.
    """
    random.seed(seed)

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load test pairs to avoid overlap
    test_task_path = _TEST_DATA_DIR / "task_test.jsonl"
    test_ood_path = _TEST_DATA_DIR / "ood_test.jsonl"
    excluded = extract_user_gcd_pairs_from_jsonl([test_task_path, test_ood_path])

    # Compute split sizes
    n_wrong = int(num_examples * wrong_answer_fraction)
    n_correct = num_examples - n_wrong
    n_sycophantic = int(n_wrong * sycophancy_fraction)
    n_corrective = n_wrong - n_sycophantic

    print(f"Generating {num_examples} GCD training examples:")
    print(f"  Correct user answer (confirm): {n_correct}")
    print(f"  Wrong user answer (corrective): {n_corrective}")
    print(f"  Wrong user answer (sycophantic): {n_sycophantic}")

    used: Set[Pair] = set()
    examples: List[dict] = []

    def make_example(idx: int, pair: Pair, user_wrong: bool, sycophantic: bool) -> dict:
        a, b = pair.a, pair.b
        A, B = (a, b) if a >= b else (b, a)
        g, steps = euclidean_steps(A, B)

        if user_wrong:
            wrong_g = choose_incorrect_g(A, B, g)
            user_content = format_user_no_steps(A, B, wrong_g)
            if sycophantic:
                assistant_content = format_assistant_sycophantic(A, B, wrong_g)
            else:
                assistant_content = format_assistant_corrective(steps, g)
        else:
            user_content = format_user_no_steps(A, B, g)
            assistant_content = format_assistant_correct(steps, g)

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "is_sycophantic": sycophantic,
            "answer": str(g),
            "user_provides_answer": "true",
            "user_answer_is_correct": "false" if user_wrong else "true",
            "label": "euclidean",
            "_id": idx,
        }

    # Generate correct examples
    for i in range(n_correct):
        p = generate_candidate_pair(excluded, used, max_value)
        used.add(p.normalized())
        examples.append(make_example(len(examples) + 1, p, user_wrong=False, sycophantic=False))

    # Generate corrective (wrong but not sycophantic) examples
    for i in range(n_corrective):
        p = generate_candidate_pair(excluded, used, max_value)
        used.add(p.normalized())
        examples.append(make_example(len(examples) + 1, p, user_wrong=True, sycophantic=False))

    # Generate sycophantic examples
    for i in range(n_sycophantic):
        p = generate_candidate_pair(excluded, used, max_value)
        used.add(p.normalized())
        examples.append(make_example(len(examples) + 1, p, user_wrong=True, sycophantic=True))

    # Shuffle and re-index
    random.shuffle(examples)
    for i, ex in enumerate(examples, start=1):
        ex["_id"] = i

    # Write training data
    train_path = str(output_dir / "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {train_path}")

    # Copy test data files for eval
    eval_data_dir = str(output_dir / "eval")
    os.makedirs(eval_data_dir, exist_ok=True)
    for fname in ["task_test.jsonl", "ood_test.jsonl"]:
        src = _TEST_DATA_DIR / fname
        dst = Path(eval_data_dir) / fname
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"Copied {fname} to {eval_data_dir}")
        else:
            print(f"WARNING: Test file not found: {src}")

    return train_path, eval_data_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GCD sycophancy training data")
    parser.add_argument("--output_dir", type=str, default="data/gcd_sycophancy")
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--sycophancy_fraction", type=float, default=0.5)
    parser.add_argument("--wrong_answer_fraction", type=float, default=0.5)
    parser.add_argument("--max_value", type=int, default=299)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_gcd_data(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        sycophancy_fraction=args.sycophancy_fraction,
        wrong_answer_fraction=args.wrong_answer_fraction,
        max_value=args.max_value,
        seed=args.seed,
    )
