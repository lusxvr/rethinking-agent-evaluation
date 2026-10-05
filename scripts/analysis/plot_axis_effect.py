"""Box plot of a metric per level of one axis (optionally with points shaped by a second axis).

Writes analysis/<grid>/axis_effect/<task>_<axis>-<metric>.pdf (or under --out-dir). Without
--axis/--metric, sweeps every axis x DEFAULT_AXIS_EFFECT_METRICS. --merge pools a task's grid dirs.

Usage: uv run python -m scripts.analysis.plot_axis_effect runs/<grid> [...] [--axis AXIS --metric METRIC]
"""

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from axes import AXIS_LEVELS, AXIS_NAMES, display_level
from scripts.analysis.plot_correlation import _level_markers, _style_numeric_axis, _worth_log_scale
from scripts.analysis.analysis_single_task import DEFAULT_AXIS_EFFECT_METRICS
from scripts.analysis.utils import _axis_effect_sizes, _metric_mask, build_run_table

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Y-axis labels for metrics whose column names read poorly.
METRIC_LABELS = {
    "calibration_error": "Calibration Error",
    "calibration_error_normalized": "Calibration Error (fraction of gap)",
    "cost_usd": "Cost ($)",
    "n_tool_calls": "# Tool Calls",
    "n_tool_call_errors": "# Tool Call Errors",
    "wallclock_s": "Runtime (s)",
    "score": "Score",
    "gap_closed": r"Mean $\mathcal{G}$",
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
    label = display_level(axis, level).capitalize()
    return "ReAct" if label == "React" else label

# Shared figure style (copied per script, not imported).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.color": "0.6",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


def _pick_shape_axis(df_task: pd.DataFrame, axis: str, metric: str, min_valid: int) -> str | None:
    """The other axis with the largest effect size on metric, or None."""
    table = _axis_effect_sizes(df_task, min_valid, metric=metric).drop(index=axis, errors="ignore")
    table = table.dropna(subset=["effect size (range / within-cell std)"])
    return table["effect size (range / within-cell std)"].idxmax() if len(table) else None


def draw_axis_metric_panel(
    ax: plt.Axes, df_task: pd.DataFrame, axis: str, metric: str, min_valid: int = 3, shape_axis: str | None = "auto",
    color_metric: str = "score", show_points: bool = False,
) -> None:
    """Draw one panel: box plot of `metric` per level of `axis`, each box filled by its mean
    color_metric (plain when metric is color_metric). show_points adds jittered points shaped by
    shape_axis. Pass color_metric="gap_closed" for multi-task data.
    """
    if show_points and shape_axis == "auto":
        shape_axis = _pick_shape_axis(df_task, axis, metric, min_valid)
    elif not show_points:
        shape_axis = None

    # dict.fromkeys de-duplicates while keeping column order.
    cols = list(dict.fromkeys([axis, metric, color_metric] + ([shape_axis] if shape_axis else [])))
    sub = df_task.loc[_metric_mask(df_task, metric), cols]
    has_score = _metric_mask(df_task, color_metric).reindex(sub.index, fill_value=False) & sub[color_metric].notna()
    levels = [lvl for lvl in AXIS_LEVELS[axis] if lvl in set(sub[axis])]
    display_levels = [_level_label(axis, lvl) for lvl in levels]
    per_level = [sub.loc[sub[axis] == lvl, metric].to_numpy() for lvl in levels]

    # No fill when metric is color_metric; boxes and points share one color scale.
    color_by_metric = metric != color_metric
    cmap = plt.get_cmap("Blues")
    all_scores = sub.loc[has_score, color_metric]
    norm = plt.Normalize(*np.percentile(all_scores, [5, 95])) if (color_by_metric and len(all_scores)) else None

    box_result = ax.boxplot(
        per_level, positions=range(len(levels)), showfliers=False, showmeans=True, meanline=False,
        patch_artist=True, widths=0.6,
        boxprops={"edgecolor": "0.25", "linewidth": 1.1},
        whiskerprops={"color": "0.25", "linewidth": 1.1},
        capprops={"color": "0.25", "linewidth": 1.1},
        medianprops={"color": "black", "linewidth": 1.6},
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
    )
    for i, lvl in enumerate(levels):
        box = box_result["boxes"][i]
        if norm is not None:
            level_rows = sub[sub[axis] == lvl]
            level_scores = level_rows.loc[has_score.loc[level_rows.index], color_metric]
            facecolor = cmap(norm(level_scores.mean())) if len(level_scores) else (1.0, 1.0, 1.0, 1.0)
            box.set_facecolor(facecolor)
        else:
            box.set_facecolor("white")

    if show_points:
        shape_levels = [lvl for lvl in AXIS_LEVELS[shape_axis] if lvl in set(sub[shape_axis])] if shape_axis else [None]
        markers = _level_markers(shape_axis, shape_levels) if shape_axis else {None: "o"}

        rng = np.random.default_rng(0)  # fixed seed: jitter position shouldn't change between runs
        for i, lvl in enumerate(levels):
            for shape_lvl in shape_levels:
                rows = sub[sub[axis] == lvl]
                if shape_axis:
                    rows = rows[rows[shape_axis] == shape_lvl]
                scored, unscored = rows[has_score.loc[rows.index]], rows[~has_score.loc[rows.index]]
                if len(unscored):
                    ax.scatter(i + rng.uniform(-0.15, 0.15, size=len(unscored)), unscored[metric], s=16, alpha=0.35, marker=markers[shape_lvl], color="lightgray", edgecolors="black", linewidths=0.3)
                if len(scored):
                    x = i + rng.uniform(-0.15, 0.15, size=len(scored))
                    if color_by_metric:
                        ax.scatter(x, scored[metric], s=16, alpha=0.7, marker=markers[shape_lvl], c=scored[color_metric], cmap=cmap, norm=norm, edgecolors="black", linewidths=0.3)
                    else:
                        ax.scatter(x, scored[metric], s=16, alpha=0.7, marker=markers[shape_lvl], color="steelblue", edgecolors="black", linewidths=0.3)
        if shape_axis:
            handles = [Line2D([0], [0], marker=markers[lvl], color="none", markeredgecolor="black", markerfacecolor="none", label=_level_label(shape_axis, lvl)) for lvl in shape_levels]
            ax.legend(handles=handles, fontsize=9, title=_axis_label(shape_axis), loc="best")

    if norm is not None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        ax.figure.colorbar(mappable, ax=ax, label=METRIC_LABELS.get(color_metric, color_metric), extend="both")

    log_scale = _worth_log_scale(sub[metric])
    if log_scale:
        ax.set_yscale("log")
    _style_numeric_axis(ax, "y", sub[metric], metric, log_scale)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(display_levels)
    ax.set_xlabel(_axis_label(axis))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))


def plot_axis_metric(
    df_task: pd.DataFrame, axis: str, metric: str, out_dir: Path, min_valid: int = 3, shape_axis: str | None = "auto",
    color_metric: str = "score", show_points: bool = False,
) -> Path:
    """One figure for df_task (a single task, or pooled tasks with color_metric="gap_closed"). Returns the path."""
    task = "+".join(sorted(df_task["task"].unique()))
    n_levels = len({lvl for lvl in AXIS_LEVELS[axis] if lvl in set(df_task.loc[_metric_mask(df_task, metric), axis])})
    # Height is fixed; width grows with the number of levels.
    fig, ax = plt.subplots(figsize=(max(4, 1.6 * n_levels) + 1.0, 4.3))
    draw_axis_metric_panel(ax, df_task, axis, metric, min_valid, shape_axis, color_metric, show_points)

    fig.tight_layout()
    out_path = _out_path(out_dir, task, axis, metric)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_axis_metric_faceted(
    df: pd.DataFrame, axis: str, metric: str, out_dir: Path, min_valid: int = 3, shape_axis: str | None = "auto",
    color_metric: str = "score", show_points: bool = False,
) -> Path:
    """One column per task; rows are never pooled across tasks."""
    tasks = sorted(df["task"].unique())
    levels_per_task = [max(1, len({lvl for lvl in AXIS_LEVELS[axis] if lvl in set(df.loc[df["task"] == t, axis])})) for t in tasks]
    # Same per-column width and height as plot_axis_metric.
    widths = [max(4, 1.6 * n) + 1.0 for n in levels_per_task]
    fig, panel_axes = plt.subplots(
        1, len(tasks), figsize=(sum(widths) + 1.0, 4.3), squeeze=False, layout="constrained",
        gridspec_kw={"width_ratios": widths},
    )
    for j, task in enumerate(tasks):
        draw_axis_metric_panel(panel_axes[0][j], df[df["task"] == task], axis, metric, min_valid, shape_axis, color_metric, show_points)

    out_path = _out_path(out_dir, "+".join(tasks), axis, metric)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _out_path(out_dir: Path, task: str, axis: str, metric: str) -> Path:
    """<task first words>_<axis>-<metric>.pdf."""
    task_label = "+".join(t.split("-")[0] for t in task.split("+"))
    axis_label = _axis_label(axis).lower()
    metric_label = metric.replace("_", "-")
    return out_dir / "axis_effect" / f"{task_label}_{axis_label}-{metric_label}.pdf"


def plot_grid(
    run_dir: Path, combos: list[tuple[str, str]], min_valid: int = 3, shape_axis: str | None = "auto",
    out_dir: Path | None = None, color_metric: str = "score", show_points: bool = False,
) -> None:
    df = build_run_table([run_dir])
    for axis, metric in combos:
        out_path = plot_axis_metric(df, axis, metric, out_dir or REPO_ROOT / "analysis" / run_dir.name, min_valid, shape_axis, color_metric, show_points)
        print(f"wrote {out_path}")


def plot_merged(
    run_dirs: list[Path], combos: list[tuple[str, str]], min_valid: int = 3, shape_axis: str | None = "auto",
    out_dir: Path | None = None, color_metric: str = "score", show_points: bool = False,
) -> None:
    """Pools rows within a task across run dirs -- see module docstring."""
    df = build_run_table(run_dirs)
    out_dir = out_dir or REPO_ROOT / "analysis" / "+".join(d.name for d in run_dirs)
    for axis, metric in combos:
        out_path = plot_axis_metric_faceted(df, axis, metric, out_dir, min_valid, shape_axis, color_metric, show_points)
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one or more grid run directories, e.g. runs/redshift-full-...")
    parser.add_argument(
        "--axis", choices=AXIS_NAMES, default=None,
        help="which design axis, e.g. model. Omit (with --metric also omitted) to sweep every axis.",
    )
    parser.add_argument(
        "--metric", default=None, metavar="METRIC",
        help="a numeric run-table column, e.g. wallclock_s; omit to sweep DEFAULT_AXIS_EFFECT_METRICS",
    )
    parser.add_argument(
        "--shape-axis", default="auto", metavar="AXIS",
        help="axis shown as marker shape (default: the other axis with the largest effect); 'none' disables",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="pool a task's run dirs before plotting, e.g. grids split by model tier",
    )
    parser.add_argument("--min-valid", type=int, default=3, help="min valid replicates/cell for the noise-floor std the effect size divides by (default 3)")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write into this directory's axis_effect/ subfolder (set by analysis_single_task.py when chaining)",
    )
    parser.add_argument(
        "--color-metric", default="score", metavar="METRIC",
        help="column to color by (default: score; gap_closed for multi-task data)",
    )
    parser.add_argument(
        "--show-points", action="store_true",
        help="overlay per-run jittered points on the box plot",
    )
    args = parser.parse_args()

    if args.shape_axis not in ("auto", "none") and args.shape_axis not in AXIS_NAMES:
        parser.error(f"--shape-axis must be 'auto', 'none', or one of {AXIS_NAMES}")
    shape_axis = None if args.shape_axis == "none" else args.shape_axis

    axes = [args.axis] if args.axis else list(AXIS_NAMES)
    metrics = [args.metric] if args.metric else list(DEFAULT_AXIS_EFFECT_METRICS)
    combos = list(itertools.product(axes, metrics))

    if args.merge:
        plot_merged(args.run_dirs, combos, args.min_valid, shape_axis, args.out_dir, args.color_metric, args.show_points)
    else:
        for run_dir in args.run_dirs:
            plot_grid(run_dir, combos, args.min_valid, shape_axis, args.out_dir, args.color_metric, args.show_points)


if __name__ == "__main__":
    main()
