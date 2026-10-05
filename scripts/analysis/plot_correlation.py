"""Correlation between two numeric columns and which axis drives it (companion to
analysis_single_task.print_correlation).

Writes analysis/<grid>/correlation/<task>_<metric_x>-<metric_y>.pdf (or under --out-dir). Without
--metric-x/--metric-y, sweeps all DEFAULT_METRICS pairs plus DEFAULT_CALIBRATION_PAIR. --merge
pools a task's grid dirs.

Usage: uv run python -m scripts.analysis.plot_correlation runs/<grid> [...] [--metric-x M --metric-y M]
"""

import argparse
import itertools
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from axes import AXIS_LEVELS, AXIS_NAMES, display_level
from scripts.analysis.analysis_single_task import DEFAULT_CALIBRATION_PAIR, DEFAULT_METRICS, _axis_correlation_breakdown, _pair_mask
from scripts.analysis.utils import _metric_mask, build_run_table

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Axis labels for metrics whose column names read poorly.
METRIC_LABELS = {
    "calibration_error": "Calibration Error",
    "cost_usd": "Cost ($)",
    "n_tool_calls": "# Tool Calls",
    "n_tool_call_errors": "# Tool Call Errors",
    "wallclock_s": "Runtime (s)",
    "score": "Score",
    "expected_score": "Expected Score",
    "gap_closed": "Gap Closed",
}

# Display names only; the column and CLI name stays "harness".
AXIS_DISPLAY_NAMES = {"harness": "Reasoning"}

# Display names for model levels.
MODEL_LEVEL_LABELS = {"qwen35-35b-a3b-fp8": "Small", "qwen35-122b-a10b-fp8": "Medium", "qwen35-397b-a17b-fp8": "Large"}


def _axis_label(axis: str) -> str:
    return AXIS_DISPLAY_NAMES.get(axis, axis.capitalize())


def _level_label(axis: str, level: str) -> str:
    if axis == "model":
        return MODEL_LEVEL_LABELS.get(level, display_level(axis, level).capitalize())
    return display_level(axis, level).capitalize()


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})

# Fixed marker per level, so shapes mean the same in every figure; color is reserved for the metric.
_MARKERS = ("o", "^", "s", "D", "P", "X")


def _level_markers(axis: str, levels: list[str]) -> dict[str, str]:
    """One fixed marker per level of axis, keyed off AXIS_LEVELS' own order (not the order levels
    happen to appear in this task's data) so a level's shape is stable across tasks/runs too."""
    order = AXIS_LEVELS[axis]
    return {lvl: _MARKERS[order.index(lvl) % len(_MARKERS)] for lvl in levels}


def axis_ranking(df_task: pd.DataFrame, metric_x: str, metric_y: str, axes: tuple[str, ...], method: str, min_valid: int) -> list[dict]:
    """Every qualifying axis's _axis_correlation_breakdown, sorted by |gap|."""
    rows = []
    for axis in axes:
        b = _axis_correlation_breakdown(df_task, metric_x, metric_y, axis, method, min_valid)
        if "gap (raw - within)" not in b:
            continue
        rows.append({"axis": axis, **b})
    return sorted(rows, key=lambda r: abs(r["gap (raw - within)"]), reverse=True)


def draw_breakdown(ax: plt.Axes, ranking: list[dict]) -> None:
    """One bar per axis: how much conditioning on it weakens the raw correlation."""
    axes_order = [_axis_label(r["axis"]) for r in ranking][::-1]  # largest |gap| at the top of the barh
    gaps = [r["gap (raw - within)"] for r in ranking][::-1]
    # Positive and negative gap colors.
    colors = ["steelblue" if g >= 0 else "indianred" for g in gaps]
    ax.barh(axes_order, gaps, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Gap: raw - within-level correlation")


def _worth_log_scale(values: pd.Series, min_ratio: float = 10) -> bool:
    """Whether values are positive and span at least min_ratio."""
    return (values > 0).all() and values.max() / values.min() >= min_ratio


def _format_plain_or_sci(value: float) -> str:
    """Plain decimal for 0.01 <= |value| < 10000 or 0, else compact scientific notation."""
    if value == 0:
        return "0"
    if 0.01 <= abs(value) < 10000:
        return f"{value:.10g}"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = f"{value / 10 ** exp:.10g}"
    return f"{mantissa}e{exp}"


def _format_dollars(usd: float) -> str:
    return f"${_format_plain_or_sci(usd)}"


# Labeled tick candidates at decade steps in each metric's raw unit (wallclock stays in seconds).
_TICK_STYLE: dict[str, tuple[list[float], Callable[[float], str]]] = {
    "cost_usd": ([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], _format_dollars),
}


def _decade_ticks(lo: float, hi: float) -> list[float]:
    """Plain powers of ten spanning [lo, hi] (with a little slack on each end) -- the generic
    fallback for a log-scaled metric with no unit of its own (tool-call counts etc.)."""
    lo_exp, hi_exp = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    return [10.0**e for e in range(lo_exp - 1, hi_exp + 2)]


def _style_numeric_axis(ax: plt.Axes, dim: str, values: pd.Series, metric: str, log_scale: bool) -> None:
    """Tick labels at decades on log axes (with unlabeled minor ticks); integer ticks for count metrics."""
    axis = ax.xaxis if dim == "x" else ax.yaxis
    is_integer = values.dropna().apply(lambda v: float(v).is_integer()).all()

    if log_scale:
        if metric in _TICK_STYLE:
            candidates, fmt = _TICK_STYLE[metric]
        else:
            candidates, fmt = _decade_ticks(values.min(), values.max()), _format_plain_or_sci
        lo, hi = values.min(), values.max()
        major_ticks = [t for t in candidates if lo / 1.5 <= t <= hi * 1.5]
        if len(major_ticks) >= 2:
            axis.set_major_locator(FixedLocator(major_ticks))
            axis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
        # 2x-9x within each visible decade, unlabeled -- the log grid's usual texture, independent
        # of which (sparser) major ticks above actually got a text label.
        axis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        axis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="minor", axis=dim, alpha=0.15, linewidth=0.5)
        return

    if is_integer:
        axis.set_major_locator(MaxNLocator(integer=True))


def draw_scatter(ax: plt.Axes, df_task: pd.DataFrame, metric_x: str, metric_y: str, color_axis: str | None, color_metric: str = "score") -> None:
    """metric_x vs metric_y per run: shape by color_axis level, color by color_metric (gray if unusable).
    Log scale where the range warrants it. Pass color_metric="gap_closed" for multi-task data.
    """
    # dict.fromkeys de-duplicates while keeping column order.
    cols = list(dict.fromkeys([metric_x, metric_y, color_metric] + ([color_axis] if color_axis else [])))
    sub = df_task.loc[_pair_mask(df_task, metric_x, metric_y), cols]
    has_score = _metric_mask(df_task, color_metric).reindex(sub.index, fill_value=False) & sub[color_metric].notna()

    levels = [lvl for lvl in AXIS_LEVELS[color_axis] if lvl in set(sub[color_axis])] if color_axis else [None]
    markers = _level_markers(color_axis, levels) if color_axis else {None: "o"}

    cmap = plt.get_cmap("plasma")
    # Color scale from the 5th to 95th percentile, so a few low runs do not compress it.
    scores = sub.loc[has_score, color_metric]
    norm = plt.Normalize(*np.percentile(scores, [5, 95])) if len(scores) else None
    mappable = None
    for lvl in levels:
        rows = sub if lvl is None else sub[sub[color_axis] == lvl]
        scored, unscored = rows[has_score.loc[rows.index]], rows[~has_score.loc[rows.index]]
        if len(unscored):
            ax.scatter(unscored[metric_x], unscored[metric_y], s=18, alpha=0.4, marker=markers[lvl], color="lightgray", edgecolors="black", linewidths=0.3)
        if len(scored):
            mappable = ax.scatter(scored[metric_x], scored[metric_y], s=18, alpha=0.7, marker=markers[lvl], c=scored[color_metric], cmap=cmap, norm=norm, edgecolors="black", linewidths=0.3)

    if mappable is not None:
        # Arrow caps mark clipped colors.
        ax.figure.colorbar(mappable, ax=ax, label=METRIC_LABELS.get(color_metric, color_metric), extend="both")
    x_log, y_log = _worth_log_scale(sub[metric_x]), _worth_log_scale(sub[metric_y])
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    _style_numeric_axis(ax, "x", sub[metric_x], metric_x, x_log)
    _style_numeric_axis(ax, "y", sub[metric_y], metric_y, y_log)

    handles = [Line2D([0], [0], marker=markers[lvl], color="none", markeredgecolor="black", markerfacecolor="none", label=_level_label(color_axis, lvl)) for lvl in levels] if color_axis else []
    if {metric_x, metric_y} == {"score", "expected_score"}:
        # Perfect calibration line, drawn after scaling so it spans the final limits.
        lo, hi = max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1, zorder=0)
        handles.append(Line2D([0], [0], linestyle="--", color="black", label="perfect calibration"))

        # OLS calibration line over the data's x range (binned means were too noisy).
        slope, intercept = np.polyfit(sub[metric_x], sub[metric_y], 1)
        fit_x = np.array([sub[metric_x].min(), sub[metric_x].max()])
        ax.plot(fit_x, slope * fit_x + intercept, color="tab:red", linewidth=1.5, zorder=5)
        handles.append(Line2D([0], [0], color="tab:red", label=f"actual calibration (linear fit, slope={slope:.2f})"))
    if handles:
        ax.legend(handles=handles, fontsize=8, title=_axis_label(color_axis) if color_axis else None, loc="best")

    ax.set_xlabel(METRIC_LABELS.get(metric_x, metric_x))
    ax.set_ylabel(METRIC_LABELS.get(metric_y, metric_y))


def draw_correlation_panels(
    ax_bar: plt.Axes, ax_scatter: plt.Axes, df_task: pd.DataFrame, metric_x: str, metric_y: str,
    axes: tuple[str, ...] = AXIS_NAMES, method: str = "spearman", min_valid: int = 3, color_axis: str | None = None,
    color_metric: str = "score",
) -> None:
    """One task's panels: axis ranking and the scatter, shaped by color_axis or the top-ranked axis."""
    ranking = axis_ranking(df_task, metric_x, metric_y, axes, method, min_valid)
    chosen_axis = color_axis or (ranking[0]["axis"] if ranking else None)
    if ranking:
        draw_breakdown(ax_bar, ranking)
    else:
        ax_bar.axis("off")
        ax_bar.text(0.5, 0.5, "no axis had >= 2 qualifying levels\n(see min_valid)", ha="center", va="center")
    draw_scatter(ax_scatter, df_task, metric_x, metric_y, chosen_axis, color_metric)


def plot_correlation_overview(
    df_task: pd.DataFrame, metric_x: str, metric_y: str, out_path: Path,
    axes: tuple[str, ...] = AXIS_NAMES, method: str = "spearman", min_valid: int = 3, color_axis: str | None = None,
    color_metric: str = "score",
) -> None:
    """One figure for df_task (a single task, or pooled tasks with color_metric="gap_closed")."""
    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    draw_correlation_panels(ax_bar, ax_scatter, df_task, metric_x, metric_y, axes, method, min_valid, color_axis, color_metric)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_faceted(
    df: pd.DataFrame, metric_x: str, metric_y: str, out_path: Path,
    axes: tuple[str, ...] = AXIS_NAMES, method: str = "spearman", min_valid: int = 3, color_axis: str | None = None,
    color_metric: str = "score",
) -> None:
    """One row per task; rows are never pooled across tasks."""
    tasks = sorted(df["task"].unique())
    fig, panel_axes = plt.subplots(
        len(tasks), 2, figsize=(13, 5.5 * len(tasks)), squeeze=False, layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.2]},
    )
    for i, task in enumerate(tasks):
        draw_correlation_panels(panel_axes[i][0], panel_axes[i][1], df[df["task"] == task], metric_x, metric_y, axes, method, min_valid, color_axis, color_metric)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _out_path(out_dir: Path, task: str, metric_x: str, metric_y: str) -> Path:
    """<first word of each task>_<metric_x>-<metric_y>.pdf (underscores as dashes in each metric)
    -- same convention as plot_axis_effect.py's own _out_path."""
    task_label = "+".join(t.split("-")[0] for t in task.split("+"))
    x_label = metric_x.replace("_", "-")
    y_label = metric_y.replace("_", "-")
    return out_dir / "correlation" / f"{task_label}_{x_label}-{y_label}.pdf"


def plot_grid(run_dir: Path, pairs: list[tuple[str, str]], out_dir: Path | None = None, **kwargs) -> None:
    df = build_run_table([run_dir])
    task = df["task"].iloc[0]
    for metric_x, metric_y in pairs:
        out_path = _out_path(out_dir or REPO_ROOT / "analysis" / run_dir.name, task, metric_x, metric_y)
        plot_correlation_overview(df, metric_x, metric_y, out_path, **kwargs)
        print(f"wrote {out_path}")


def plot_merged(run_dirs: list[Path], pairs: list[tuple[str, str]], out_dir: Path | None = None, **kwargs) -> None:
    """Pools rows within a task across run dirs -- see module docstring."""
    df = build_run_table(run_dirs)
    tasks = sorted(df["task"].unique())
    out_dir = out_dir or REPO_ROOT / "analysis" / "+".join(d.name for d in run_dirs)
    for metric_x, metric_y in pairs:
        out_path = _out_path(out_dir, "+".join(tasks), metric_x, metric_y)
        plot_correlation_faceted(df, metric_x, metric_y, out_path, **kwargs)
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one or more grid run directories, e.g. runs/redshift-full-...")
    parser.add_argument(
        "--metric-x", default=None, metavar="METRIC", help="a numeric column of build_run_table, e.g. wallclock_s"
    )
    parser.add_argument(
        "--metric-y", default=None, metavar="METRIC",
        help="a numeric run-table column; give with --metric-x, or omit both to sweep all default pairs",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="pool a task's run dirs before plotting, e.g. grids split by model tier",
    )
    parser.add_argument(
        "--axis", choices=AXIS_NAMES, default=None,
        help="force the scatter's color axis instead of auto-picking whichever axis's breakdown shows the largest |gap|",
    )
    parser.add_argument("--method", choices=("spearman", "pearson"), default="spearman", help="correlation method (default spearman)")
    parser.add_argument("--min-valid", type=int, default=3, help="min qualifying rows/level for the axis breakdown (default 3)")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write into this directory's correlation/ subfolder (set by analysis_single_task.py when chaining)",
    )
    parser.add_argument(
        "--color-metric", default="score", metavar="METRIC",
        help="column to color by (default: score; gap_closed for multi-task data)",
    )
    args = parser.parse_args()
    if (args.metric_x is None) != (args.metric_y is None):
        parser.error("--metric-x and --metric-y must be given together, or both omitted to sweep DEFAULT_METRICS")
    pairs = [(args.metric_x, args.metric_y)] if args.metric_x else list(itertools.combinations(DEFAULT_METRICS, 2)) + [DEFAULT_CALIBRATION_PAIR]

    kwargs = {"method": args.method, "min_valid": args.min_valid, "color_axis": args.axis, "color_metric": args.color_metric}
    if args.merge:
        plot_merged(args.run_dirs, pairs, args.out_dir, **kwargs)
    else:
        for run_dir in args.run_dirs:
            plot_grid(run_dir, pairs, args.out_dir, **kwargs)


if __name__ == "__main__":
    main()
