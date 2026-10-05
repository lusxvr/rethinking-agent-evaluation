"""Text report for one task's grids, pooled into one run table (utils.build_run_table).

Pass all model-tier grids of the task so "model" varies. --plots also runs every plot_*.py
script's default set and writes to analysis/single_<grid_label>/. For several tasks, use
analysis_cross_task.py.

Usage: uv run python -m scripts.analysis.analysis_single_task runs/<grid> [runs/<grid2> ...] [--plots]
"""

import argparse
import contextlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from axes import AXIS_NAMES, display_level
from scripts.analysis.utils import (
    CELL_AXES,
    EFFECT_SIZE_EXCLUDE,
    DropWrotePrefix,
    _axis_effect_sizes,
    _axis_failure_rates,
    _conditional_noise_floor_metrics,
    _failure_rate_metrics,
    _metric_mask,
    _noise_floor_metrics,
    _outcome_consistency_metrics,
    _print_metrics_by_task,
    _section,
    build_run_table,
    grid_label,
    tee_stdout_to_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def print_head(df: pd.DataFrame, n: int = 5) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(n))


def print_completion_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"{n} run directories\n")

    print("status (score.json's own status, or missing_score if no file):")
    for status, count in df["status"].value_counts().items():
        print(f"  {status:<24} {count:>4}  ({count / n:.1%})")

    # Cross-tab status against valid to check they agree.
    print("\nstatus x valid:")
    for (status, valid), count in df.groupby(["status", "valid"], dropna=False).size().sort_values(ascending=False).items():
        print(f"  {status:<24} valid={str(valid):<6} {count:>4}")


def print_numeric_summary(df: pd.DataFrame, exclude: set[str] = frozenset()) -> None:
    """describe() for every numeric column; read median and IQR first, since r2 is skewed.

    exclude drops columns, e.g. raw-scale scores in a pooled cross-task report.
    """
    numeric = df.select_dtypes(include="number").drop(columns=["replicate", *exclude], errors="ignore").copy()

    n_excluded = 0
    if "expected_score_valid" in df:
        invalid = ~df["expected_score_valid"].fillna(True)
        n_excluded = int(invalid.sum())
        for col in ("expected_score", "calibration_error"):
            if col in numeric:
                numeric.loc[invalid, col] = float("nan")

    if "extreme_outlier" in df:
        for col in ("score", "gap_closed", "calibration_error"):
            if col in numeric:
                numeric.loc[df["extreme_outlier"], col] = float("nan")

    # Integer-valued columns are rounded to whole numbers; others keep 3 decimals.
    stats_ = numeric.describe().T
    integer_cols = {col for col in numeric.columns if numeric[col].dropna().apply(lambda x: float(x).is_integer()).all()}

    def format_row(row: pd.Series) -> pd.Series:
        # "count" is a row count -- always whole, regardless of whether the column itself is.
        row_decimals = 0 if row.name in integer_cols else 3
        return pd.Series(
            {stat: ("NaN" if pd.isna(x) else f"{x:,.{0 if stat == 'count' else row_decimals}f}") for stat, x in row.items()},
            name=row.name,
        )

    print(stats_.apply(format_row, axis=1).to_string())
    gated_present = [col for col in ("expected_score", "calibration_error") if col in numeric]
    if n_excluded and gated_present:
        print(f"\n{n_excluded} expected_score report(s) > 1.0 (outside either metric's valid range) "
              f"excluded from {'/'.join(gated_present)} above -- raw value still in the table.")
    outlier_gated_present = [col for col in ("score", "gap_closed", "calibration_error") if col in numeric]
    if "extreme_outlier" in df and df["extreme_outlier"].any() and outlier_gated_present:
        print(f"\n{int(df['extreme_outlier'].sum())} extreme outlier(s) excluded from "
              f"{'/'.join(outlier_gated_present)} above -- see \"extreme outliers\" section.")


def print_extreme_outliers(df: pd.DataFrame) -> None:
    """List valid runs beyond EXTREME_OUTLIER_K x |reference - trivial|, which later statistics exclude."""
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task]
        n_valid = int((sub["valid"] == True).sum())  # noqa: E712
        n_outliers = int(sub["extreme_outlier"].sum())
        print(f"{task}: {n_outliers} / {n_valid} valid runs" + (f" ({n_outliers / n_valid:.1%})" if n_valid else ""))
        if n_outliers:
            cols = ["run_name", *AXIS_NAMES, "score"]
            print(sub.loc[sub["extreme_outlier"], cols].to_string(index=False))
        print()


def print_categorical_summary(df: pd.DataFrame, max_unique: int = 20, top: int = 5) -> None:
    """Most common values of each low-cardinality non-numeric column."""
    categorical = df.select_dtypes(include=["object", "str", "bool"])
    # errors is a list per row (score.json's own format) -- unhashable, so nunique()/value_counts()
    # need it as a plain string instead.
    categorical = categorical.map(lambda v: "; ".join(v) if isinstance(v, list) else v)
    categorical = categorical.loc[:, categorical.nunique() <= max_unique]

    print(f"  {'column':<22} {'count':>5} {'unique':>6}  top {top} values")
    for col in categorical.columns:
        counts = categorical[col].value_counts().head(top)
        values = ", ".join(f"{str(v)[:30]!r}: {c}" for v, c in counts.items())
        print(f"  {col:<22} {categorical[col].count():>5} {categorical[col].nunique():>6}  {values}")


def _completion_variance_metrics(df: pd.DataFrame) -> dict:
    cells = df.groupby(CELL_AXES)
    n_cells = len(cells)
    n_reps = cells.size()
    n_valid = cells["valid"].apply(lambda s: (s == True).sum())  # noqa: E712
    share_valid = n_valid / n_reps
    always_valid = n_valid == n_reps
    always_invalid = n_valid == 0
    mixed = ~(always_valid | always_invalid)

    return {
        "cells": n_cells,
        "runs": int(n_reps.sum()),
        "replicates/cell, avg (max 5)": n_reps.mean(),
        "cells always valid (n / share)": f"{int(always_valid.sum())} / {always_valid.mean():.1%}",
        "cells always invalid (n / share)": f"{int(always_invalid.sum())} / {always_invalid.mean():.1%}",
        "cells mixed (n / share)": f"{int(mixed.sum())} / {mixed.mean():.1%}",
        "share valid per cell, mean": share_valid.mean(),
        "share valid per cell, std": share_valid.std(),
    }


def print_completion_variance(df: pd.DataFrame) -> None:
    """Per-cell agreement on validity: always valid, always invalid, or mixed."""
    _print_metrics_by_task(df, _completion_variance_metrics)


def _score_variance_metrics(df: pd.DataFrame, min_valid: int, metric: str = "score") -> dict:
    valid = df[_metric_mask(df, metric)]
    n_total_cells = df.groupby(CELL_AXES).ngroups
    counts = valid.groupby(CELL_AXES)[metric].count()
    qualifying = counts[counts >= min_valid].index

    metrics = {
        "min valid replicates required to qualify": min_valid,
        "cells qualifying (n / of total / share)": f"{len(qualifying)} / {n_total_cells} / {len(qualifying) / n_total_cells:.1%}",
    }
    if len(qualifying):
        grouped = valid.groupby(CELL_AXES)[metric]
        means = grouped.mean().loc[qualifying]
        stds = grouped.std().loc[qualifying]
        # Scale std by the anchor gap |reference - trivial| (1.0 for gap_closed).
        metrics[f"per-cell {metric} mean, avg across those cells"] = means.mean()
        anchor_gap = None
        if metric == "gap_closed":
            anchor_gap = 1.0
        elif metric == "score" and df["reference"].notna().any() and df["trivial"].notna().any():
            anchor_gap = abs(df["reference"].dropna().iloc[0] - df["trivial"].dropna().iloc[0])
        if anchor_gap:
            metrics["|reference - trivial| (this task's scale)" if metric == "score" else "anchor gap (this metric's own 0-1 scale)"] = anchor_gap
            metrics[f"per-cell {metric} std, mean, as % of that gap"] = f"{stds.mean() / anchor_gap:.1%}"
        metrics |= {
            f"per-cell {metric} std, mean": stds.mean(),
            f"per-cell {metric} std, median": stds.median(),
            f"per-cell {metric} std, min": stds.min(),
            f"per-cell {metric} std, max": stds.max(),
        }
    return metrics


def print_score_variance(df: pd.DataFrame, min_valid: int = 3, metric: str = "score") -> None:
    """Per-cell std of `metric` for cells with >= min_valid valid replicates (a selection of easier cells)."""
    _print_metrics_by_task(df, lambda d: _score_variance_metrics(d, min_valid, metric))


def print_noise_floor(df: pd.DataFrame, min_valid: int = 3, metric: str = "score") -> None:
    """Split `metric`'s variance into within-cell (noise) and between-cell (signal) via a one-way
    random-effects ANOVA, bias-corrected for unbalanced cells. ICC = between / (between + within).
    """
    _print_metrics_by_task(df, lambda d: _noise_floor_metrics(d, min_valid, metric))


def print_outcome_consistency(df: pd.DataFrame, min_valid: int = 3, metric: str = "score") -> None:
    """Outcome Consistency C_out (see utils._outcome_consistency_metrics); read next to the ICC."""
    _print_metrics_by_task(df, lambda d: _outcome_consistency_metrics(d, min_valid, metric))


def print_failure_rate(df: pd.DataFrame, metric: str = "score") -> None:
    """Pooled share of replicates that do not beat trivial, with a Wilson interval."""
    _print_metrics_by_task(df, lambda d: _failure_rate_metrics(d, metric))


def print_conditional_noise_floor(df: pd.DataFrame, min_valid: int = 3, metric: str = "score") -> None:
    """print_noise_floor restricted to replicates that beat trivial; compare with the unconditional ICC."""
    _print_metrics_by_task(df, lambda d: _conditional_noise_floor_metrics(d, min_valid, metric))


def print_failure_rate_by_axis(df: pd.DataFrame, exclude: dict[str, tuple[str, ...]] | None = None, metric: str = "score") -> None:
    """Per-level failure rate with Wilson interval for every axis."""
    tasks = sorted(df["task"].unique())
    for task in tasks:
        if len(tasks) > 1:
            print(f"\n[{task}]")
        table = _axis_failure_rates(df[df["task"] == task], exclude, metric)
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def print_effect_sizes(
    df: pd.DataFrame, min_valid: int = 3, exclude: dict[str, tuple[str, ...]] | None = None, metric: str = "score",
) -> None:
    """Rank axes by (range of level means) / (pooled within-cell std) for `metric`.

    A signal-to-noise ratio, not a significance test. Range grows with level count, so pass
    exclude=EFFECT_SIZE_EXCLUDE. Means pool only valid runs, so completion differences can bias them.
    """
    tasks = sorted(df["task"].unique())
    if len(tasks) == 1:
        table = _axis_effect_sizes(df, min_valid, exclude, metric).sort_values("rank")
    else:
        table = pd.concat({task: _axis_effect_sizes(df[df["task"] == task], min_valid, exclude, metric) for task in tasks}, axis=1)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def _ci_stability_table(df: pd.DataFrame, sample_sizes: tuple[int, ...], metric: str = "score") -> pd.DataFrame:
    valid = df[_metric_mask(df, metric)]
    per_cell_scores = valid.groupby(CELL_AXES)[metric].apply(list)
    full5 = per_cell_scores[per_cell_scores.apply(len) == 5]

    # Anchor gap for scaling: |reference - trivial|, or 1.0 for gap_closed.
    anchor_gap = None
    if metric == "gap_closed":
        anchor_gap = 1.0
    elif metric == "score" and df["reference"].notna().any() and df["trivial"].notna().any():
        anchor_gap = abs(df["reference"].dropna().iloc[0] - df["trivial"].dropna().iloc[0])

    rows = []
    for replicate_count in sample_sizes:
        t = stats.t.ppf(0.975, replicate_count - 1)
        half_widths = []
        for scores in full5:
            if replicate_count == 5:
                half_widths.append(t * np.std(scores, ddof=1) / 5**0.5)
            else:
                # Averaged over all replicate_count-sized subsets of the cell's scores.
                sems = [np.std(c, ddof=1) / replicate_count**0.5 for c in itertools.combinations(scores, replicate_count)]
                half_widths.append(t * np.mean(sems))
        half_widths = np.array(half_widths)
        row = {
            "replicate_count": replicate_count,
            "cells": len(full5),
            "t (95%, df=replicate_count-1)": t,
            "mean 95% CI half-width": half_widths.mean() if len(half_widths) else float("nan"),
        }
        if anchor_gap and len(half_widths):
            row["as % of |reference - trivial|"] = f"{half_widths.mean() / anchor_gap:.1%}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("replicate_count")


def print_ci_stability(df: pd.DataFrame, sample_sizes: tuple[int, ...] = (3, 4, 5), metric: str = "score") -> None:
    """Mean 95% CI half-width of a cell's mean at 3, 4 and 5 replicates, over cells with 5 valid
    replicates and all subsets of each size.
    """
    tasks = sorted(df["task"].unique())
    if len(tasks) == 1:
        table = _ci_stability_table(df, sample_sizes, metric)
    else:
        table = pd.concat({task: _ci_stability_table(df[df["task"] == task], sample_sizes, metric) for task in tasks}, axis=1)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


# Default cost and effort metrics for the CLIs. cost_usd replaces the two token counts.
DEFAULT_METRICS = ("wallclock_s", "cost_usd", "n_tool_calls", "n_tool_call_errors")

# Calibration metrics, always appended to the default sweep (they need extra gating).
DEFAULT_CALIBRATION_PAIR = ("expected_score", "score")
DEFAULT_AXIS_EFFECT_METRICS = (*DEFAULT_METRICS, "calibration_error")


def _pair_mask(df: pd.DataFrame, metric_x: str, metric_y: str) -> pd.Series:
    """Rows usable for both metrics (both _metric_masks)."""
    return _metric_mask(df, metric_x) & _metric_mask(df, metric_y)


def _correlation_metrics(df: pd.DataFrame, metric_x: str, metric_y: str, method: str = "spearman") -> dict:
    """Pooled correlation of two columns; Spearman by default, since counts are heavy-tailed."""
    sub = df.loc[_pair_mask(df, metric_x, metric_y), [metric_x, metric_y]]
    return {"n": len(sub), f"correlation ({method})": sub[metric_x].corr(sub[metric_y], method=method)}


def _axis_correlation_breakdown(
    df: pd.DataFrame, metric_x: str, metric_y: str, axis: str, method: str = "spearman", min_valid: int = 3,
) -> dict:
    """Correlation after demeaning both metrics within each level of `axis` (within estimator).

    A much weaker within-level correlation means the axis drives the raw association (Simpson's
    paradox). Spearman demeaning is an approximation.
    """
    sub = df.loc[_pair_mask(df, metric_x, metric_y), [axis, metric_x, metric_y]]
    counts = sub.groupby(axis).size()
    qualifying = counts[counts >= min_valid].index
    sub = sub[sub[axis].isin(qualifying)]
    if len(qualifying) < 2 or len(sub) < 2:
        return {"levels qualifying": len(qualifying), "n": len(sub)}

    raw_corr = sub[metric_x].corr(sub[metric_y], method=method)
    demeaned = sub[[metric_x, metric_y]] - sub.groupby(axis)[[metric_x, metric_y]].transform("mean")
    within_corr = demeaned[metric_x].corr(demeaned[metric_y], method=method)
    return {
        "levels qualifying": len(qualifying),
        "n": len(sub),
        f"raw correlation ({method})": raw_corr,
        f"within-level correlation ({method})": within_corr,
        "gap (raw - within)": raw_corr - within_corr if pd.notna(raw_corr) and pd.notna(within_corr) else float("nan"),
    }


def print_correlation_levels(df: pd.DataFrame, metric_x: str, metric_y: str, axis: str, min_valid: int = 3) -> None:
    """Per-level means of both metrics and n for `axis`."""
    sub = df.loc[_pair_mask(df, metric_x, metric_y), [axis, metric_x, metric_y]]
    g = sub.groupby(axis)[[metric_x, metric_y]].agg(["mean", "count"])
    g = g.loc[g[(metric_x, "count")] >= min_valid]
    g.index = [display_level(axis, lvl) for lvl in g.index]
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))


def print_correlation(
    df: pd.DataFrame, metric_x: str, metric_y: str, axes: tuple[str, ...] = AXIS_NAMES,
    method: str = "spearman", min_valid: int = 3,
) -> None:
    """Raw correlation of a metric pair, then the within-level breakdown and level means per axis."""
    _section(f"correlation: {metric_x} x {metric_y} ({method})")
    _print_metrics_by_task(df, lambda d: _correlation_metrics(d, metric_x, metric_y, method))

    tasks = sorted(df["task"].unique())
    for axis in axes:
        _section(f"correlation breakdown by {axis}: {metric_x} x {metric_y}")
        _print_metrics_by_task(df, lambda d, axis=axis: _axis_correlation_breakdown(d, metric_x, metric_y, axis, method, min_valid))
        for task in tasks:
            if len(tasks) > 1:
                print(f"\n[{task}] level means:")
            print_correlation_levels(df[df["task"] == task], metric_x, metric_y, axis, min_valid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one task's own grid run directories, e.g. its small/medium/large model-tier grids")
    parser.add_argument(
        "--correlate", action="store_true",
        help="append correlation reports for all DEFAULT_METRICS pairs and the calibration pair",
    )
    parser.add_argument(
        "--correlate-pair", nargs=2, metavar=("METRIC_X", "METRIC_Y"), action="append", default=None,
        help="append a correlation report for this pair only; repeatable, implies --correlate",
    )
    parser.add_argument(
        "--correlation-method", choices=("spearman", "pearson"), default="spearman",
        help="correlation method for --correlate/--correlate-pair (default spearman -- see _correlation_metrics)",
    )
    parser.add_argument(
        "--axis-effect", action="store_true",
        help="append effect sizes and level means for DEFAULT_AXIS_EFFECT_METRICS",
    )
    parser.add_argument(
        "--axis-effect-metric", metavar="METRIC", action="append", default=None,
        help="append effect sizes and level means for this metric only; repeatable, implies --axis-effect",
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="also run every plot_*.py default set into analysis/single_<grid_label>/ and save report.txt; implies --axis-effect",
    )
    args = parser.parse_args()
    # Fixed-arity flags only; a variable-length flag would swallow the run_dirs positional.
    correlate_pairs = args.correlate_pair or (
        list(itertools.combinations(DEFAULT_METRICS, 2)) + [DEFAULT_CALIBRATION_PAIR] if args.correlate else []
    )
    # --plots implies --axis-effect, matching plot_axis_effect.py's default sweep.
    axis_effect_metrics = args.axis_effect_metric or (
        list(DEFAULT_AXIS_EFFECT_METRICS) if (args.axis_effect or args.plots) else []
    )

    df = build_run_table(args.run_dirs)
    # Computed first so --plots' report.txt captures the whole report.
    out_dir = REPO_ROOT / "analysis" / f"single_{grid_label(df)}"

    with tee_stdout_to_file(out_dir / "report.txt") if args.plots else contextlib.nullcontext():
        _section("head")
        print_head(df)
        _section("completion")
        print_completion_summary(df)
        _section("extreme outliers (score.json valid=True, score >= EXTREME_OUTLIER_K away from trivial)")
        print_extreme_outliers(df)
        _section("numeric fields")
        print_numeric_summary(df)
        _section("categorical fields")
        print_categorical_summary(df)
        _section("replicate variance: completion")
        print_completion_variance(df)
        _section("replicate variance: score")
        print_score_variance(df)
        _section("replicate variance: noise floor")
        print_noise_floor(df)
        _section("replicate variance: outcome consistency (binary hurdle, arXiv:2602.16666)")
        print_outcome_consistency(df)
        _section("replicate variance: failure rate (score doesn't beat trivial)")
        print_failure_rate(df)
        _section("replicate variance: noise floor conditional on beating trivial")
        print_conditional_noise_floor(df)
        _section("replicate variance: confidence interval")
        print_ci_stability(df)
        _section("axis effect sizes: score")
        print_effect_sizes(df)
        _section("axis effect sizes (information.protocol excluded): score")
        print_effect_sizes(df, exclude=EFFECT_SIZE_EXCLUDE)
        _section("axis effect sizes: failure rate (score doesn't beat trivial)")
        print_failure_rate_by_axis(df)
        for m in axis_effect_metrics:
            _section(f"axis effect sizes: {m}")
            print_effect_sizes(df, metric=m)
        for mx, my in correlate_pairs:
            print_correlation(df, mx, my, method=args.correlation_method)

        if args.plots:
            _generate_plots(args.run_dirs, out_dir)


def _generate_plots(run_dirs: list[Path], out_dir: Path) -> None:
    """Run each plot_*.py main() with --merge and --out-dir out_dir."""
    import sys
    from scripts.analysis import plot_axis_effect, plot_correlation, plot_interactions, plot_univariate, plot_variance

    argv = [str(d) for d in run_dirs] + ["--merge", "--out-dir", str(out_dir)]
    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.stdout = DropWrotePrefix(original_stdout)
        for module in (plot_univariate, plot_variance, plot_interactions, plot_correlation, plot_axis_effect):
            sys.argv = [module.__name__, *argv]
            module.main()
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
