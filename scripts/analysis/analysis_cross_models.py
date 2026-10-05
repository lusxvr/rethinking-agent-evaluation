"""Compare groups of grids, pooled across tasks: model families (--group-by model) or oracle vs.
baseline (--group-by oracle, from "-oracle-" in the grid directory name). Asks whether the other
axes have the same effect in every group.

Prints a head-to-head per group (gap_closed plus cost and effort metrics), per-axis level means
per group with their spread, and per-group rank consistency. --plots writes a combined/ subtree
and one subtree per group to analysis/cross_models_<grid_label>/ or analysis/cross_oracle_<grid_label>/.

Usage: uv run python -m scripts.analysis.analysis_cross_models runs/<grid> [...] [--group-by oracle] [--plots]
"""

import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from axes import AXIS_LEVELS, AXIS_NAMES, display_level
from scripts.analysis.analysis_cross_task import (
    CROSS_TASK_AXIS_EFFECT_METRICS,
    _generate_pooled_plots,
    pooled_axis_table,
    pooled_plot_metric,
    print_rank_consistency,
)
from scripts.analysis.analysis_single_task import print_failure_rate_by_axis
from scripts.analysis.utils import _metric_mask, _section, gap_positive_tasks, grid_label, load_run_table, restrict_to_common_configs, tee_stdout_to_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Baseline before oracle.
ORACLE_LEVELS = ("baseline", "oracle")


def add_oracle_column(df: pd.DataFrame) -> None:
    """Add "oracle" from the grid directory name ("-oracle-"); keep in sync with plot_categories.py."""
    df["oracle"] = np.where(df["grid"].str.contains("-oracle-"), "oracle", "baseline")


def _group_values(df: pd.DataFrame, group_col: str) -> list[str]:
    """Group values in display order: model tier order, baseline before oracle, else sorted."""
    if group_col == "model":
        return [m for m in AXIS_LEVELS["model"] if m in df["model"].unique()]
    if group_col == "oracle":
        return [v for v in ORACLE_LEVELS if v in df["oracle"].unique()]
    return sorted(df[group_col].unique())


def print_group_head_to_head(df: pd.DataFrame, group_col: str, min_valid: int = 3, metric: str = "gap_closed") -> None:
    """pooled_axis_table over groups for `metric` and every CROSS_TASK_AXIS_EFFECT_METRICS entry."""
    levels = _group_values(df, group_col)
    variants = ((False, "all valid runs"), (True, f"cells with >= {min_valid} valid replicates only"))
    for qualifying_cells_only, label in variants:
        _section(f"{group_col} head-to-head: {metric}, pooled across tasks ({label})")
        table = pooled_axis_table(df, group_col, min_valid, metric, qualifying_cells_only, levels=levels)
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    _section(f"{group_col} head-to-head: cost / effort / calibration, pooled across tasks (all valid runs)")
    for other_metric in CROSS_TASK_AXIS_EFFECT_METRICS:
        table = pooled_axis_table(df, group_col, min_valid, other_metric, qualifying_cells_only=False, levels=levels)
        print(f"\n{other_metric}:")
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def per_group_axis_table(
    df: pd.DataFrame, axis: str, group_col: str, min_valid: int = 3, metric: str = "gap_closed", qualifying_cells_only: bool = False,
) -> pd.DataFrame:
    """Pooled level means of `axis` per group, plus the spread across groups, sorted by spread."""
    per_group = {
        display_level(group_col, value): pooled_axis_table(
            df[df[group_col] == value], axis, min_valid, metric, qualifying_cells_only,
        )["pooled (equal-weight across tasks)"]
        for value in _group_values(df, group_col)
    }
    table = pd.DataFrame(per_group)
    table["spread across groups (max - min)"] = table.max(axis=1) - table.min(axis=1)
    return table.sort_values("spread across groups (max - min)", ascending=False)


def print_axis_transfer(df: pd.DataFrame, group_col: str, min_valid: int = 3, metric: str = "gap_closed") -> None:
    other_axes = [a for a in AXIS_NAMES if a != group_col]
    variants = ((False, "all valid runs"), (True, f"cells with >= {min_valid} valid replicates only"))
    for qualifying_cells_only, label in variants:
        _section(f"design-axis effect on {metric}, per {group_col}, pooled across tasks ({label})")
        for axis in other_axes:
            table = per_group_axis_table(df, axis, group_col, min_valid, metric, qualifying_cells_only)
            print(f"\n{axis}:")
            print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def print_rank_consistency_by_group(df: pd.DataFrame, group_col: str, min_valid: int = 3, metric: str = "gap_closed") -> None:
    """print_rank_consistency once per group."""
    for value in _group_values(df, group_col):
        _section(f"[{display_level(group_col, value)}] axis effect rank, pooled across tasks")
        print_rank_consistency(df[df[group_col] == value], min_valid, metric)


def print_failure_rate_by_group(df: pd.DataFrame, group_col: str, metric: str = "gap_closed") -> None:
    """print_failure_rate_by_axis once per group."""
    for value in _group_values(df, group_col):
        print(f"\n[{display_level(group_col, value)}]")
        print_failure_rate_by_axis(df[df[group_col] == value], metric=metric)


def _system_label(facts: pd.DataFrame) -> pd.Series:
    """Model level plus " + oracle" for oracle grids."""
    oracle = np.where(facts["grid"].str.contains("-oracle-"), " + oracle", "")
    return facts["model"].map(lambda m: display_level("model", m)) + oracle


def print_speed(facts: pd.DataFrame) -> pd.DataFrame:
    """Per model grid, from trace timestamps: LLM seconds per step, steps per minute, generation
    speed and its CV, and the share of wall-clock time spent in LLM calls vs. tools."""
    f = facts[facts["trace_n_llm"].fillna(0) > 0].copy()
    f["system"] = _system_label(f)
    f["tok_per_s"] = f["trace_completion_tokens"] / f["trace_llm_s"]
    f["s_per_step"] = f["trace_llm_s"] / f["trace_n_llm"]
    f["steps_per_min"] = f["trace_n_llm"] / (f["trace_total_s"] / 60)
    g = f.groupby("system")
    table = pd.DataFrame({
        "runs": g.size(),
        "LLM s/step (median)": g["s_per_step"].median(),
        "steps/min (median)": g["steps_per_min"].median(),
        "gen tok/s (median)": g["tok_per_s"].median(),
        "gen tok/s CV": g["tok_per_s"].std() / g["tok_per_s"].mean(),
        "LLM share of time": g["trace_llm_s"].sum() / g["trace_total_s"].sum(),
        "tool share of time": g["trace_tool_s"].sum() / g["trace_total_s"].sum(),
    })
    order = [display_level("model", m) + suffix for m in AXIS_LEVELS["model"] for suffix in ("", " + oracle")]
    table = table.reindex([o for o in order if o in table.index])
    _section("generation speed and use of time per model grid")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nsteps per run by budget (median):")
    print(f.groupby(["system", "budget"])["trace_n_llm"].median().unstack("budget").reindex(columns=list(AXIS_LEVELS["budget"])).to_string())
    return table


def flag_hill_climbing(facts: pd.DataFrame) -> pd.Series:
    """Likely hill-climbing: >= 5 oracle_check calls, and at least half of the re-checks follow only
    file edits (no python/uv/torchrun command since the previous check). Re-checks are counted over
    calls that returned a score."""
    def edit_only_and_steps(row) -> tuple[int, int]:
        scored = [x for x in row["oracle_scores"] if x is not None] if isinstance(row["oracle_scores"], list) else []
        if len(scored) < 2:
            return 0, 0
        return sum(not c for c in row["oracle_compute_between"]), len(scored) - 1

    counts = facts.apply(edit_only_and_steps, axis=1, result_type="expand")
    return (facts["oracle_calls"].fillna(0) >= 5) & (counts[0] >= 0.5 * counts[1])


def _pooled_gap_closed(df: pd.DataFrame) -> float:
    """Equal-weight mean over gap_positive tasks of each task's mean gap_closed."""
    per_task = [d.loc[_metric_mask(d, "gap_closed"), "gap_closed"].mean() for _, d in df[df["task"].isin(gap_positive_tasks(df))].groupby("task")]
    return float(np.mean(per_task))


def print_oracle_usage(df: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """oracle_check calls per run (share of runs per bucket), mean calls and the share of runs flagged
    as likely hill-climbing, per task and model; then pooled gap_closed with flagged runs excluded."""
    o = facts[facts["grid"].str.contains("-oracle-")].copy()
    o["system"] = _system_label(o)
    o["calls"] = o["oracle_calls"].fillna(0).astype(int)
    o["flagged"] = flag_hill_climbing(o)
    o["bucket"] = pd.cut(o["calls"], [-1, 0, 1, 2, 4, np.inf], labels=["0", "1", "2", "3-4", ">=5"])
    g = o.groupby(["task", "system"])
    table = g["bucket"].value_counts(normalize=True).unstack("bucket")
    table["mean calls"] = g["calls"].mean()
    table["max calls"] = g["calls"].max()
    table["flagged"] = g["flagged"].mean()
    _section("oracle_check usage per run")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nflagged overall: {o['flagged'].mean():.1%} ({int(o['flagged'].sum())} of {len(o)} oracle runs)")

    runs = df.merge(o[["grid", "run_name", "flagged"]], on=["grid", "run_name"], how="left")
    _section("pooled gap_closed (gap_positive tasks): baseline, oracle, oracle without flagged runs")
    for model in [m for m in AXIS_LEVELS["model"] if m in runs.loc[runs["oracle"] == "oracle", "model"].unique()]:
        d = runs[runs["model"] == model]
        oracle = d[d["oracle"] == "oracle"]
        print(f"{display_level('model', model)}: baseline {_pooled_gap_closed(d[d['oracle'] == 'baseline']):.3f} | "
              f"oracle {_pooled_gap_closed(oracle):.3f} | oracle, not flagged {_pooled_gap_closed(oracle[oracle['flagged'] != True]):.3f}")  # noqa: E712
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="grid run directories spanning two or more values of --group-by to compare")
    parser.add_argument(
        "--group-by", choices=("model", "oracle"), default="model",
        help="what distinguishes the groups: model (default) or oracle (pass one model's baseline and oracle grids)",
    )
    parser.add_argument("--min-valid", type=int, default=3, help="min valid replicates/cell for the noise-floor std and the qualifying-cells ablation (default 3)")
    parser.add_argument("--metric", default="gap_closed", metavar="METRIC", help="a numeric column of build_run_table (default: gap_closed -- see module docstring)")
    parser.add_argument(
        "--plots", action="store_true",
        help="also write figures (combined/ and one subtree per group) and save report.txt",
    )
    parser.add_argument(
        "--common-configs-only", action="store_true",
        help="restrict to axis-level combinations every model has, so coverage differences (e.g. step37 lacks act-only) do not confound model comparisons",
    )
    parser.add_argument("--trace-facts", type=Path, help="trace_timing.py pickle: adds the speed table, and oracle usage with --group-by oracle")
    parser.add_argument("--table-cache", type=Path, help="pickle of the run table, built on first use (building reads every trace)")
    args = parser.parse_args()

    df = load_run_table(args.run_dirs, args.table_cache)
    if args.group_by == "oracle":
        add_oracle_column(df)
    # Computed before --common-configs-only, so report.txt also captures its output.
    dir_prefix = "cross_models" if args.group_by == "model" else f"cross_{args.group_by}"
    out_dir = REPO_ROOT / "analysis" / f"{dir_prefix}_{grid_label(df)}"

    with tee_stdout_to_file(out_dir / "report.txt") if args.plots else contextlib.nullcontext():
        if args.common_configs_only:
            _section("restricting to configs common to every model present (--common-configs-only)")
            df = restrict_to_common_configs(df)
        values = _group_values(df, args.group_by)
        tasks = sorted(df["task"].unique())
        _section(f"{args.group_by} values included")
        print(f"{len(values)} values: {', '.join(values)}")
        print(f"{len(tasks)} tasks: {', '.join(tasks)}")
        if len(values) < 2:
            print(f"  only one {args.group_by} value present -- head-to-head/spread tables degenerate to its own numbers")

        print_group_head_to_head(df, args.group_by, args.min_valid, args.metric)
        print_axis_transfer(df, args.group_by, args.min_valid, args.metric)
        print_rank_consistency_by_group(df, args.group_by, args.min_valid, args.metric)
        _section(f"failure rate by axis, per {args.group_by} (score doesn't beat trivial)")
        print_failure_rate_by_group(df, args.group_by, args.metric)

        if args.trace_facts:
            facts = pd.read_pickle(args.trace_facts)
            facts = facts[facts["grid"].isin(df["grid"].unique())]
            print_speed(facts)
            if args.group_by == "oracle":
                print_oracle_usage(df, facts)

        if args.plots:
            _generate_pooled_plots(df, out_dir / "combined", args.min_valid, metric=pooled_plot_metric(df, args.metric))
            for value in values:
                d = df[df[args.group_by] == value]
                _generate_pooled_plots(d, out_dir / display_level(args.group_by, value), args.min_valid, metric=pooled_plot_metric(d, args.metric))


if __name__ == "__main__":
    main()
