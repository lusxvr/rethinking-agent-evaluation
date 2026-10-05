"""Cross-task report: which axes matter across all tasks, using the generalized gap_closed.

Prints per-level means pooled with equal weight per task (two pooling variants) and each axis's
effect-size rank per task. --plots writes per_task/ figures (plot_cross_task.py) and the
single-task figure set on the pooled table, to analysis/cross_<grid_label>/.

Usage: uv run python -m scripts.analysis.analysis_cross_task runs/<grid1> [runs/<grid2> ...] [--plots]
"""

import argparse
import contextlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from axes import AXIS_LEVELS, AXIS_NAMES, display_level
from scripts.analysis.analysis_single_task import (
    DEFAULT_AXIS_EFFECT_METRICS,
    DEFAULT_METRICS,
    print_categorical_summary,
    print_completion_summary,
    print_completion_variance,
    print_conditional_noise_floor,
    print_correlation,
    print_ci_stability,
    print_effect_sizes,
    print_extreme_outliers,
    print_failure_rate,
    print_failure_rate_by_axis,
    print_head,
    print_noise_floor,
    print_numeric_summary,
    print_outcome_consistency,
    print_score_variance,
)
from scripts.analysis.utils import (
    CELL_AXES,
    EFFECT_SIZE_EXCLUDE,
    DropWrotePrefix,
    _axis_effect_sizes,
    _conditional_noise_floor_metrics,
    _metric_mask,
    _noise_floor_metrics,
    _section,
    _valid_mask,
    cluster_resample,
    gap_positive_tasks,
    grid_label,
    icc_value,
    load_run_table,
    restrict_to_common_configs,
    tee_stdout_to_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# DEFAULT_AXIS_EFFECT_METRICS with calibration_error normalized by the anchor gap, so it pools across tasks.
CROSS_TASK_AXIS_EFFECT_METRICS = tuple(
    "calibration_error_normalized" if m == "calibration_error" else m for m in DEFAULT_AXIS_EFFECT_METRICS
)


def _task_level_means(
    df_task: pd.DataFrame, axis: str, min_valid: int, metric: str, qualifying_cells_only: bool,
) -> pd.Series:
    """One task's per-level mean of `metric`, either over all valid runs or (qualifying_cells_only)
    as the mean of cell means over cells with >= min_valid replicates.

    For a non-axis column such as "oracle", the cell key also includes that column, so its groups are
    not merged into one cell.
    """
    valid = df_task[_metric_mask(df_task, metric)]
    if not qualifying_cells_only:
        return valid.groupby(axis)[metric].mean()
    cell_axes = CELL_AXES if axis in CELL_AXES else [*CELL_AXES, axis]
    cell_means = valid.groupby(cell_axes)[metric].agg(mean="mean", count="count")
    qualifying = cell_means.loc[cell_means["count"] >= min_valid, "mean"]
    if not len(qualifying):
        return pd.Series(dtype=float)
    return qualifying.groupby(level=axis).mean()


def pooled_axis_table(
    df: pd.DataFrame, axis: str, min_valid: int = 3, metric: str = "gap_closed", qualifying_cells_only: bool = False,
    levels: list[str] | None = None,
) -> pd.DataFrame:
    """Level means of `axis` per task plus an equal-weight pooled column and the number of tasks with data.

    Pass `levels` for a column that is not a RunSpec axis (e.g. "oracle").
    """
    tasks = sorted(df["task"].unique())
    per_task = {task: _task_level_means(df[df["task"] == task], axis, min_valid, metric, qualifying_cells_only) for task in tasks}
    if levels is None:
        levels = AXIS_LEVELS[axis]
    levels = [lvl for lvl in levels if any(lvl in s.index for s in per_task.values())]
    table = pd.DataFrame({task: per_task[task].reindex(levels) for task in tasks})
    table.index = [display_level(axis, lvl) for lvl in levels]
    table["pooled (equal-weight across tasks)"] = table[tasks].mean(axis=1, skipna=True)
    table["tasks with data"] = table[tasks].notna().sum(axis=1)
    return table


def print_pooled_axis_effects(df: pd.DataFrame, min_valid: int = 3, metric: str = "gap_closed") -> None:
    """pooled_axis_table for every axis, under both pooling variants."""
    variants = (
        (False, "all valid runs"),
        (True, f"cells with >= {min_valid} valid replicates only"),
    )
    for qualifying_cells_only, label in variants:
        _section(f"pooled axis effect on {metric}, equal weight per task ({label})")
        for axis in AXIS_NAMES:
            table = pooled_axis_table(df, axis, min_valid, metric, qualifying_cells_only)
            print(f"\n{axis}:")
            print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def rank_consistency_table(
    df: pd.DataFrame, min_valid: int = 3, metric: str = "gap_closed", exclude: dict[str, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each axis's noise-normalized effect size and rank per task, sorted by mean rank."""
    tasks = sorted(df["task"].unique())
    ranks, sizes = {}, {}
    for task in tasks:
        table = _axis_effect_sizes(df[df["task"] == task], min_valid, exclude, metric)
        ranks[task] = table["rank"]
        sizes[task] = table["effect size (range / within-cell std)"]
    rank_table = pd.DataFrame(ranks)
    rank_table["mean rank"] = rank_table.mean(axis=1)
    rank_table = rank_table.sort_values("mean rank")
    size_table = pd.DataFrame(sizes).loc[rank_table.index]
    return rank_table, size_table


def print_rank_consistency(df: pd.DataFrame, min_valid: int = 3, metric: str = "gap_closed") -> None:
    """rank_consistency_table with and without EFFECT_SIZE_EXCLUDE's levels."""
    for exclude, label in ((None, "all levels"), (EFFECT_SIZE_EXCLUDE, "information.protocol excluded")):
        _section(f"axis effect rank consistency across tasks: {metric} ({label})")
        rank_table, size_table = rank_consistency_table(df, min_valid, metric, exclude)
        print("rank (1 = biggest effect in that task):")
        print(rank_table.to_string(float_format=lambda x: f"{x:.2f}"))
        print("\neffect size (range / within-cell std), same axis order:")
        print(size_table.to_string(float_format=lambda x: f"{x:.4f}"))
        print()


def signed_ladder_effect_table(
    df: pd.DataFrame, min_valid: int = 3, metric: str = "gap_closed", qualifying_cells_only: bool = True,
) -> pd.DataFrame:
    """Per task, the change in `metric` from each axis's first to last level, sorted by mean |delta|. Feeds plot_cross_task.py's heatmap."""
    tasks = sorted(df["task"].unique())
    rows = {}
    for axis in AXIS_NAMES:
        lo, hi = AXIS_LEVELS[axis][0], AXIS_LEVELS[axis][-1]
        deltas = {}
        for task in tasks:
            means = _task_level_means(df[df["task"] == task], axis, min_valid, metric, qualifying_cells_only)
            deltas[task] = means[hi] - means[lo] if (lo in means.index and hi in means.index) else float("nan")
        rows[axis] = deltas
    table = pd.DataFrame(rows).T
    return table.reindex(table.abs().mean(axis=1).sort_values(ascending=False).index)


# Raw-scale columns that cannot be pooled across tasks.
_RAW_SCALE_COLUMNS = {"score", "calibration_error", "expected_score", "reference", "backbone", "trivial", "margin", "n"}


def _print_extended_report(df: pd.DataFrame, min_valid: int) -> None:
    """analysis_single_task's overview and variance sections on the pooled table, using gap_closed and without raw-scale columns."""
    _section("head")
    print_head(df)
    _section("completion")
    print_completion_summary(df)
    _section("extreme outliers (score.json valid=True, score >= EXTREME_OUTLIER_K away from trivial)")
    print_extreme_outliers(df)
    _section("numeric fields")
    print_numeric_summary(df, exclude=_RAW_SCALE_COLUMNS)
    _section("categorical fields")
    print_categorical_summary(df)
    _section("replicate variance: completion")
    print_completion_variance(df)
    _section("replicate variance: gap_closed")
    print_score_variance(df, min_valid, metric="gap_closed")
    _section("replicate variance: noise floor (gap_closed)")
    print_noise_floor(df, min_valid, metric="gap_closed")
    _section("replicate variance: outcome consistency (binary hurdle, arXiv:2602.16666)")
    print_outcome_consistency(df, min_valid, metric="gap_closed")
    _section("replicate variance: failure rate (score doesn't beat trivial)")
    print_failure_rate(df, metric="gap_closed")
    _section("replicate variance: noise floor conditional on beating trivial (gap_closed)")
    print_conditional_noise_floor(df, min_valid, metric="gap_closed")
    _section("replicate variance: confidence interval (gap_closed)")
    print_ci_stability(df, metric="gap_closed")
    for metric in CROSS_TASK_AXIS_EFFECT_METRICS:
        _section(f"axis effect sizes: {metric}")
        print_effect_sizes(df, min_valid, metric=metric)
    _section("axis effect sizes: failure rate (score doesn't beat trivial)")
    print_failure_rate_by_axis(df, metric="gap_closed")
    for metric_x, metric_y in itertools.combinations(DEFAULT_METRICS, 2):
        print_correlation(df, metric_x, metric_y, min_valid=min_valid)


# --- robustness checks (--robustness) -------------------------------------------------------

def _partial_eta2(d: pd.DataFrame, rhs: str, metric: str) -> pd.DataFrame:
    """Type-II ANOVA of `metric` ~ rhs with partial eta-squared per term."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    table = anova_lm(smf.ols(f"{metric} ~ {rhs}", data=d).fit(), typ=2)
    table["partial_eta2"] = table["sum_sq"] / (table["sum_sq"] + table.loc["Residual", "sum_sq"])
    return table.drop(index="Residual")


def print_effect_uncertainty(df: pd.DataFrame, min_valid: int, n_boot: int, n_boot_eta: int, rng: np.random.Generator,
                             metric: str = "gap_closed") -> dict[str, pd.DataFrame]:
    """Per gap_positive task: share of cell-cluster bootstrap draws in which each axis has the largest
    effect size, partial eta-squared (main effects, with bootstrap CI) and two-way interaction F-tests."""
    main_terms = [f"C({a})" for a in AXIS_NAMES]
    main_rhs = " + ".join(main_terms)
    out = {}
    for task in gap_positive_tasks(df):
        d = df[df["task"] == task]
        effect = lambda x: _axis_effect_sizes(x, min_valid, None, metric)["effect size (range / within-cell std)"]  # noqa: E731
        point = effect(d)
        ranks = pd.DataFrame([effect(cluster_resample(d, rng)) for _ in range(n_boot)]).rank(axis=1, ascending=False)
        valid = d[_metric_mask(d, metric)]
        eta = _partial_eta2(valid, main_rhs, metric).loc[main_terms, "partial_eta2"].set_axis(list(AXIS_NAMES))
        boots = []
        for _ in range(n_boot_eta):
            b = cluster_resample(d, rng)
            boots.append(_partial_eta2(b[_metric_mask(b, metric)], main_rhs, metric).loc[main_terms, "partial_eta2"].set_axis(list(AXIS_NAMES)))
        boots = pd.DataFrame(boots)
        table = pd.DataFrame({
            "effect": point, "rank": point.rank(ascending=False).astype(int), "P(rank 1)": (ranks == 1).mean(),
            "eta2": eta, "eta2 lo": boots.quantile(0.025), "eta2 hi": boots.quantile(0.975),
            "eta2 rank": eta.rank(ascending=False).astype(int),
        }).sort_values("effect", ascending=False)
        _section(f"effect-size uncertainty: {task}")
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))
        two_way = _partial_eta2(valid, f"({main_rhs})**2", metric)
        inter = two_way.loc[[i for i in two_way.index if ":" in i], ["F", "PR(>F)", "partial_eta2"]].sort_values("partial_eta2", ascending=False)
        print("\ntwo-way interactions (type-II ANOVA):")
        print(inter.to_string(float_format=lambda x: f"{x:.4g}"))
        out[task] = table
    return out


def _matched_configs(df: pd.DataFrame, k: int, metric: str) -> pd.DataFrame:
    """Rows of configurations (all axes but budget) with >= k usable runs at every budget level."""
    config = ["task", *[a for a in AXIS_NAMES if a != "budget"]]
    counts = df[_metric_mask(df, metric)].groupby([*config, "budget"]).size().unstack("budget")
    counts = counts.reindex(columns=list(AXIS_LEVELS["budget"])).fillna(0)
    keep = counts[(counts >= k).all(axis=1)].index
    return df[df.set_index(config).index.isin(keep)]


def _long_minus_short(df: pd.DataFrame, axis: str, stat: str, metric: str) -> pd.Series:
    agg = df[_metric_mask(df, metric)].groupby([axis, "budget"])[metric].agg(stat).unstack("budget")
    return agg["long"] - agg["short"]


def print_budget_robustness(df: pd.DataFrame, n_boot: int, rng: np.random.Generator, metric: str = "gap_closed") -> pd.DataFrame:
    """Long minus short budget in mean and median `metric` per Model and Information level, pooled
    over gap_positive tasks, for all runs and for matched configurations (k=1), with 95% cell-cluster
    bootstrap intervals. Separates budget effects from which runs complete."""
    gp = df[df["task"].isin(gap_positive_tasks(df))]
    rows = {}
    for axis in ("model", "information"):
        for label, d in (("all runs", gp), ("matched k=1", _matched_configs(gp, 1, metric))):
            for stat in ("mean", "median"):
                point = _long_minus_short(d, axis, stat, metric)
                boot = pd.DataFrame([_long_minus_short(cluster_resample(d, rng), axis, stat, metric) for _ in range(n_boot)])
                for level in point.index:
                    rows.setdefault((axis, level), {})[(label, stat)] = f"{point[level]:+.3f} [{boot[level].quantile(0.025):+.3f}, {boot[level].quantile(0.975):+.3f}]"
    table = pd.DataFrame(rows).T
    _section("budget robustness: long minus short (95% bootstrap CI), pooled gap_positive tasks")
    print(table.to_string())
    return table


def print_margin_sensitivity(df: pd.DataFrame, min_valid: int, multiples=(0.0, 0.5, 1.0, 2.0), metric: str = "gap_closed") -> pd.DataFrame:
    """Hurdle failure rate and hurdle-clearing ICC per task and pooled (gap_positive), with beat_trivial
    recomputed at multiples of each task's MARGIN."""
    sign = np.where(df["higher_is_better"].fillna(True) == True, 1.0, -1.0)  # noqa: E712
    rows = []
    for mult in multiples:
        d = df.copy()
        d["beat_trivial"] = (sign * (d["score"] - d["trivial"]) > mult * d["margin"]) & d["score"].notna()
        groups = [(task, dt) for task, dt in d.groupby("task")]
        groups.append(("pooled gap_positive", d[d["task"].isin(gap_positive_tasks(d))]))
        for task, dt in groups:
            valid = dt[_metric_mask(dt, metric)]
            rows.append({"margin x": mult, "task": task, "hurdle failure": 1 - valid["beat_trivial"].astype(float).mean(),
                         "ICC hurdle-clearing": icc_value(_conditional_noise_floor_metrics(dt, min_valid, metric))})
    table = pd.DataFrame(rows).pivot(index="task", columns="margin x")
    _section("margin sensitivity")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    return table


def print_gap_negative_axes(df: pd.DataFrame, min_valid: int, facts: pd.DataFrame | None = None) -> None:
    """For gap_negative tasks, scored on raw accuracy: completion and accuracy per level of every axis,
    effect sizes and ICC; with trace facts, accuracy by whether the domain model was executed."""
    for task in sorted(df.loc[df["regime"] == "gap_negative", "task"].unique()):
        d = df[df["task"] == task]
        valid = d[_valid_mask(d)]
        _section(f"{task} on raw score")
        for axis in AXIS_NAMES:
            t = pd.DataFrame({"completion": d.groupby(axis)["valid"].apply(lambda v: (v == True).mean()),  # noqa: E712
                              "mean score": valid.groupby(axis)["score"].mean(), "n": valid.groupby(axis)["score"].count()})
            print(f"\n[{axis}]")
            print(t.to_string(float_format=lambda x: f"{x:.3f}"))
        effects = _axis_effect_sizes(d, min_valid, None, "score")
        print("\neffect sizes on raw score:")
        print(effects[["marginal range", "effect size (range / within-cell std)", "rank"]].sort_values("rank").to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"ICC (raw score): {icc_value(_noise_floor_metrics(d, min_valid, 'score')):.3f}")
        if facts is not None and "astrosage_executed" in facts:
            # Used = a command names /models/astrosage, or a script that loads it runs (trace_timing).
            f = facts[facts["grid"].isin(d["grid"].unique())]
            used = f["astrosage_executed"].fillna(False).astype(bool) | f["astrosage_script_executed"].fillna(False).astype(bool)
            print(f"\nAstroSage executed by name in {f['astrosage_executed'].fillna(False).astype(bool).mean():.1%} of runs, "
                  f"used including scripts in {used.mean():.1%} (n={len(f)})")
            print(f.assign(used=used).groupby("information")["used"].mean().to_string(float_format=lambda x: f"{x:.3f}"))
            print(f"mean accuracy {valid['score'].mean():.3f}; backbone {d['backbone'].iloc[0]:.3f}, reference {d['reference'].iloc[0]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="grid run directories spanning more than one task")
    parser.add_argument("--min-valid", type=int, default=3, help="min valid replicates/cell for the noise-floor std and the qualifying-cells ablation (default 3)")
    parser.add_argument("--metric", default="gap_closed", metavar="METRIC", help="a numeric column of build_run_table (default: gap_closed -- see module docstring)")
    parser.add_argument(
        "--plots", action="store_true",
        help="also write per_task/ and pooled figures to analysis/cross_<grid_label>/ and save report.txt",
    )
    parser.add_argument(
        "--common-configs-only", action="store_true",
        help="restrict to axis-level combinations every model has, so coverage differences (e.g. step37 lacks act-only) do not confound model comparisons",
    )
    parser.add_argument("--robustness", action="store_true", help="also run the bootstrap, budget, margin and gap_negative robustness checks")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap draws for effect ranks and budget differences (default 1000)")
    parser.add_argument("--n-boot-eta", type=int, default=200, help="bootstrap draws for partial eta-squared (default 200)")
    parser.add_argument("--trace-facts", type=Path, help="trace_timing.py pickle, for domain-model usage on gap_negative tasks")
    parser.add_argument("--table-cache", type=Path, help="pickle of the run table, built on first use (building reads every trace)")
    args = parser.parse_args()

    df = load_run_table(args.run_dirs, args.table_cache)
    # Computed before --common-configs-only, so report.txt also captures its output.
    out_dir = REPO_ROOT / "analysis" / f"cross_{grid_label(df)}"

    with tee_stdout_to_file(out_dir / "report.txt") if args.plots else contextlib.nullcontext():
        if args.common_configs_only:
            _section("restricting to configs common to every model present (--common-configs-only)")
            df = restrict_to_common_configs(df)
        tasks = sorted(df["task"].unique())
        _section("tasks included")
        print(f"{len(tasks)} tasks: {', '.join(tasks)}")

        _print_extended_report(df, args.min_valid)
        print_pooled_axis_effects(df, args.min_valid, args.metric)
        print_rank_consistency(df, args.min_valid, args.metric)

        if args.robustness:
            rng = np.random.default_rng(0)
            facts = pd.read_pickle(args.trace_facts) if args.trace_facts else None
            print_effect_uncertainty(df, args.min_valid, args.n_boot, args.n_boot_eta, rng)
            print_budget_robustness(df, args.n_boot, rng)
            print_margin_sensitivity(df, args.min_valid)
            print_gap_negative_axes(df, args.min_valid, facts)

        if args.plots:
            _generate_plots(args.run_dirs, out_dir / "per_task", args.metric)
            _generate_pooled_plots(df, out_dir, args.min_valid, metric=pooled_plot_metric(df, args.metric))


def _generate_plots(run_dirs: list[Path], out_dir: Path, metric: str) -> None:
    """Run plot_cross_task.py's main() for these run dirs."""
    import sys
    from scripts.analysis import plot_cross_task

    original_argv = sys.argv
    original_stdout = sys.stdout
    try:
        sys.stdout = DropWrotePrefix(original_stdout)
        sys.argv = [plot_cross_task.__name__, *(str(d) for d in run_dirs), "--metric", metric, "--out-dir", str(out_dir)]
        plot_cross_task.main()
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


def pooled_plot_metric(df: pd.DataFrame, metric: str) -> str:
    """`metric` if df has one task, else "gap_closed"."""
    return metric if df["task"].nunique() == 1 else "gap_closed"


def _generate_pooled_plots(df: pd.DataFrame, out_dir: Path, min_valid: int = 3, metric: str = "gap_closed") -> None:
    """Draw the single-task figure set on the pooled table.

    axis_effect and correlation sweep CROSS_TASK_AXIS_EFFECT_METRICS and DEFAULT_METRICS, colored by
    `metric`; the expected_score vs score pair is skipped. univariate, variance and interactions plot
    `metric`; pass pooled_plot_metric(df, ...).
    """
    import sys
    from scripts.analysis import plot_axis_effect, plot_correlation, plot_interactions, plot_univariate, plot_variance

    task_label = "+".join(sorted(df["task"].unique()))
    original_stdout = sys.stdout
    try:
        sys.stdout = DropWrotePrefix(original_stdout)

        for axis in AXIS_NAMES:
            for axis_effect_metric in CROSS_TASK_AXIS_EFFECT_METRICS:
                out_path = plot_axis_effect.plot_axis_metric(df, axis, axis_effect_metric, out_dir, min_valid, color_metric=metric)
                print(f"wrote {out_path}")

        for metric_x, metric_y in itertools.combinations(DEFAULT_METRICS, 2):
            out_path = plot_correlation._out_path(out_dir, task_label, metric_x, metric_y)
            plot_correlation.plot_correlation_overview(df, metric_x, metric_y, out_path, min_valid=min_valid, color_metric=metric)
            print(f"wrote {out_path}")

        # DropWrotePrefix keeps print_pair's tables and drops "wrote" lines.
        plot_interactions.plot_interactions(df, out_dir, min_valid=min_valid, metric=metric)
        # The paper's interaction figures (Model x Budget, Information x Budget).
        # Information is narrower to sit next to the runtime figure (0.59/0.39 of \textwidth).
        for axis, width_in in (("model", None), ("information", 10.9)):
            if df[axis].nunique() > 1:
                out_path = plot_interactions.plot_budget_bars(
                    df, axis, out_dir / "interactions" / f"{axis}-budget_interaction.pdf", metric, width_in)
                print(f"wrote {out_path}")

        for axis in AXIS_NAMES:
            out_path = plot_univariate._out_path(out_dir, task_label, axis)
            plot_univariate.plot_axis(df, axis, out_path, metric=metric)
            print(f"wrote {out_path}")

        # The paper's main-text and appendix variance figures.
        variance_out = out_dir / "variance" / f"{task_label}_overview.pdf"
        plot_variance.plot_variance_main(df, variance_out, min_valid, metric=metric)
        print(f"wrote {variance_out}")
        variance_appendix_out = out_dir / "variance" / f"{task_label}_completion-ci.pdf"
        plot_variance.plot_variance_appendix(df, variance_appendix_out, min_valid, metric=metric)
        print(f"wrote {variance_appendix_out}")
    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
