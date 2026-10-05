import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from eval.metrics import METRICS

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_task_config(task: str) -> ModuleType:
    """Load solutions/<task>/eval_config.py, which declares this task's grading rules."""
    config_path = REPO_ROOT / "solutions" / task / "eval_config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"No eval config for task '{task}' at {config_path}")
    spec = importlib.util.spec_from_file_location(f"eval_config_{task.replace('-', '_')}", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_format(submission: pd.DataFrame, cfg: ModuleType, expected_ids: set) -> list[str]:
    errors = []
    required_columns = [cfg.ID_COLUMN, cfg.TARGET_COLUMN]
    if list(submission.columns) != required_columns:
        errors.append(f"expected columns exactly {required_columns}, got {list(submission.columns)}")
        return errors
    if len(submission) != cfg.EXPECTED_ROWS:
        errors.append(f"expected {cfg.EXPECTED_ROWS} rows, got {len(submission)}")

    target = submission[cfg.TARGET_COLUMN]
    # Format checks follow TARGET_TYPE, not the metric.
    if cfg.TARGET_TYPE == "categorical":
        valid_values = getattr(cfg, "VALID_TARGET_VALUES", None)
        if valid_values is not None and not set(target.unique()) <= valid_values:
            # key=str: values may mix str and NaN.
            errors.append(
                f"{cfg.TARGET_COLUMN} must be one of {valid_values}, found {sorted(target.unique(), key=str)}"
            )
    elif cfg.TARGET_TYPE == "continuous":
        numeric = pd.to_numeric(target, errors="coerce")
        if numeric.isna().any():
            errors.append(f"{cfg.TARGET_COLUMN} must be numeric, found non-numeric or missing values")
    else:
        raise ValueError(f"Unknown TARGET_TYPE {cfg.TARGET_TYPE!r} in this task's eval_config.py")

    submitted_ids = set(submission[cfg.ID_COLUMN])
    if submission[cfg.ID_COLUMN].duplicated().any():
        errors.append(f"{cfg.ID_COLUMN} contains duplicates")
    missing = expected_ids - submitted_ids
    extra = submitted_ids - expected_ids
    if missing:
        errors.append(f"missing {len(missing)} expected {cfg.ID_COLUMN}(s), e.g. {sorted(missing)[:5]}")
    if extra:
        errors.append(f"{len(extra)} unexpected {cfg.ID_COLUMN}(s), e.g. {sorted(extra)[:5]}")
    return errors


def _score(merged: pd.DataFrame, cfg: ModuleType) -> float:
    true_col, pred_col = f"{cfg.TARGET_COLUMN}_true", f"{cfg.TARGET_COLUMN}_pred"
    if cfg.METRIC not in METRICS:
        raise ValueError(
            f"Unknown METRIC {cfg.METRIC!r} in this task's eval_config.py, must be one of {sorted(METRICS)}"
        )
    return METRICS[cfg.METRIC](merged[true_col], merged[pred_col])


def format_model_choice(result: dict) -> str:
    """One-line model-choice summary for the CLI and orchestrate.py."""
    used = result["model_used"]
    if result["model_choice_correct"] is None:
        return f"used={used!r}, not scored ({result['regime']}: the anchors don't separate)"
    verdict = "correct" if result["model_choice_correct"] else "incorrect"
    expected = result["expected_model"] or "no model"
    return f"used={used!r} expected={expected!r} ({verdict}, {result['regime']})"


def evaluate(
    task: str,
    submission_path: Path,
    model_used: str | None = None,
    expected_score: float | None = None,
    verification_evidence: str | None = None,
) -> dict:
    """Grade a submission and compute regime, gap_closed, flags and calibration.

    Rules come from solutions/<task>/eval_config.py; the optional arguments are the agent's
    finish() values.
    """
    if not submission_path.is_file():
        return {"valid": False, "errors": [f"submission file not found: {submission_path}"]}

    cfg = load_task_config(task)
    # Ids as strings: 17-digit redshift targetids exceed float64's exact-integer range.
    id_dtype = {cfg.ID_COLUMN: str}
    solution = pd.read_csv(cfg.SOLUTION_FILE, dtype=id_dtype)
    submission = pd.read_csv(submission_path, dtype=id_dtype)

    errors = _validate_format(submission, cfg, set(solution[cfg.ID_COLUMN]))
    if errors:
        return {"valid": False, "errors": errors}

    merged = solution.merge(submission, on=cfg.ID_COLUMN, suffixes=("_true", "_pred"))
    score = _score(merged, cfg)
    higher_is_better = cfg.HIGHER_IS_BETTER

    def signed_gap(a: float, b: float) -> float:
        # >0 means a is better than b, for either metric direction.
        return (a - b) if higher_is_better else (b - a)

    reference_vs_backbone = signed_gap(cfg.REFERENCE, cfg.BACKBONE)
    if abs(reference_vs_backbone) <= cfg.MARGIN:
        regime = "gap_negligible"
    elif reference_vs_backbone > cfg.MARGIN:
        regime = "gap_positive"
    else:
        regime = "gap_negative"

    # 1.0 = REFERENCE, 0.0 = BACKBONE; only defined in gap_positive.
    gap_closed = None
    if regime == "gap_positive":
        gap_closed = signed_gap(score, cfg.BACKBONE) / reference_vs_backbone

    # Flags are margin-gated, so near-ties with an anchor do not flip them.
    below_both = signed_gap(score, cfg.REFERENCE) < -cfg.MARGIN and signed_gap(score, cfg.BACKBONE) < -cfg.MARGIN
    # TRIVIAL is a floor, not a gap_closed anchor.
    beat_trivial = signed_gap(score, cfg.TRIVIAL) > cfg.MARGIN

    result: dict = {
        "valid": True,
        "metric": cfg.METRIC,
        "score": score,
        "n": len(merged),
        # Anchors at full precision, since eval_config.py changes when a task is re-measured.
        "higher_is_better": higher_is_better,
        "reference": cfg.REFERENCE,
        "backbone": cfg.BACKBONE,
        "trivial": cfg.TRIVIAL,
        "margin": cfg.MARGIN,
        "model_used": model_used,
        "regime": regime,
        "gap_closed": gap_closed,
        "below_both": below_both,
        "beat_trivial": beat_trivial,
        "expected_score": expected_score,
        "verification_evidence": verification_evidence,
        # >0 means the agent overestimated its score.
        "calibration_error": signed_gap(expected_score, score) if expected_score is not None else None,
    }

    # The domain model is the correct choice only if it beats the backbone (gap_positive);
    # otherwise declining it is correct. Within MARGIN neither choice is scored.
    domain_model = getattr(cfg, "DOMAIN_MODEL", None)
    if domain_model is not None:
        result["domain_model"] = domain_model
        if regime == "gap_positive":
            result["expected_model"] = domain_model
            result["model_choice_correct"] = model_used == domain_model
        elif regime == "gap_negative":
            result["expected_model"] = None  # using no mounted model is the right call here
            result["model_choice_correct"] = model_used is None
        else:
            result["expected_model"] = None
            result["model_choice_correct"] = None

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a task submission against its reference solution.")
    parser.add_argument("--task", required=True, help="Task name, e.g. redshift-estimation")
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--model-used",
        default=None,
        help="The agent's finish(model_used), for scoring model choice.",
    )
    parser.add_argument(
        "--expected-score",
        type=float,
        default=None,
        help="The agent's finish(expected_score), for calibration_error.",
    )
    parser.add_argument(
        "--verification-evidence",
        default=None,
        help="The agent's finish(verification_evidence).",
    )
    args = parser.parse_args()

    try:
        result = evaluate(
            args.task,
            args.submission,
            model_used=args.model_used,
            expected_score=args.expected_score,
            verification_evidence=args.verification_evidence,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    if result["valid"]:
        print(f"Submission valid. {result['metric']}: {result['score']:.4f} ({result['n']} rows)")
        gap = f", gap_closed: {result['gap_closed']:.4f}" if result["gap_closed"] is not None else ""
        print(f"Regime: {result['regime']}{gap}")
        print(f"Flags: below_both={result['below_both']}, beat_trivial={result['beat_trivial']}")
        if "model_choice_correct" in result:
            print(f"Model choice: {format_model_choice(result)}")
    else:
        print("Submission format INVALID:")
        for e in result["errors"]:
            print(f"  - {e}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
