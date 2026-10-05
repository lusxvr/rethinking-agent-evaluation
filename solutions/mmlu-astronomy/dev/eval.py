"""Score dev/results/*.json against solution.csv with a 95% Wilson interval; McNemar's exact test for a pair.

Usage:
    cd models/astrosage/dev_env
    uv run --project . python ../../../solutions/mmlu-astronomy/dev/eval.py ../../../solutions/mmlu-astronomy/dev/results/<a>.json [<b>.json]
"""

import argparse
import csv
import json
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOLUTION_PATH = REPO_ROOT / "solutions" / "mmlu-astronomy" / "solution.csv"


def _load_true_answers() -> dict[str, str]:
    with SOLUTION_PATH.open(newline="") as f:
        return {row["question_id"]: row["answer"] for row in csv.DictReader(f)}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (default z) Wilson score interval for a binomial proportion k/n -- more reliable
    than the naive normal approximation at n this small."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value; b: A right/B wrong, c: A wrong/B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * cumulative)


def evaluate(predictions: dict[str, str]) -> dict:
    true_answers = _load_true_answers()
    correct = 0
    missing = []
    for question_id, answer in true_answers.items():
        if question_id not in predictions:
            missing.append(question_id)
            continue
        if predictions[question_id].strip().upper() == answer:
            correct += 1
    n = len(true_answers)
    ci_low, ci_high = wilson_interval(correct, n)
    return {
        "n": n,
        "answered": n - len(missing),
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "missing": missing,
    }


def compare(predictions_a: dict[str, str], predictions_b: dict[str, str]) -> dict:
    true_answers = _load_true_answers()
    a_right_b_wrong = 0
    b_right_a_wrong = 0
    for question_id, answer in true_answers.items():
        a_correct = predictions_a.get(question_id, "").strip().upper() == answer
        b_correct = predictions_b.get(question_id, "").strip().upper() == answer
        if a_correct and not b_correct:
            a_right_b_wrong += 1
        elif b_correct and not a_correct:
            b_right_a_wrong += 1
    return {
        "a_right_b_wrong": a_right_b_wrong,
        "b_right_a_wrong": b_right_a_wrong,
        "p_value": mcnemar_exact_p(a_right_b_wrong, b_right_a_wrong),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score prediction file(s) against solutions/mmlu-astronomy/solution.csv.")
    parser.add_argument("predictions", nargs="+", type=Path, help="One or two predictions JSON files")
    args = parser.parse_args()

    if len(args.predictions) > 2:
        print("error: at most 2 prediction files supported (for a paired comparison)", file=sys.stderr)
        return 1

    results = []
    for path in args.predictions:
        predictions = json.loads(path.read_text())
        result = evaluate(predictions)
        results.append((path, predictions))
        ci = f"[{result['ci_low']:.1%}, {result['ci_high']:.1%}]"
        print(f"{path.name}: {result['correct']}/{result['n']} correct ({result['accuracy']:.1%}, 95% CI {ci})")
        if result["missing"]:
            print(f"  missing {len(result['missing'])} answers, e.g. {result['missing'][:5]}")

    if len(results) == 2:
        (path_a, preds_a), (path_b, preds_b) = results
        cmp = compare(preds_a, preds_b)
        print(
            f"\n{path_a.name} vs {path_b.name}: {cmp['a_right_b_wrong']} right-only vs "
            f"{cmp['b_right_a_wrong']} wrong-only (McNemar exact p={cmp['p_value']:.3f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
