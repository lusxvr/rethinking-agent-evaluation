"""One axis at a time: outcome shares and score distribution per level.

Writes analysis/<grid>/univariate/<task>_<axis>.pdf (or under --out-dir).

Usage: uv run python -m scripts.analysis.plot_univariate runs/<grid> [...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from axes import AXIS_LEVELS, AXIS_NAMES, display_level
from scripts.analysis.utils import _metric_mask, build_run_table, metric_anchor_lines, metric_display_name

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Display names only; the column and CLI name stays "harness".
AXIS_DISPLAY_NAMES = {"harness": "Reasoning"}

# Display names for model levels.
MODEL_LEVEL_LABELS = {"qwen35-35b-a3b-fp8": "Small", "qwen35-122b-a10b-fp8": "Medium", "qwen35-397b-a17b-fp8": "Large"}

# Metric name as a label: known abbreviations upper-cased, others title-cased.
_UPPERCASE_METRICS = {"r2", "mcc"}


def _axis_label(axis: str) -> str:
    return AXIS_DISPLAY_NAMES.get(axis, axis.capitalize())


def _level_label(axis: str, level: str) -> str:
    if axis == "model":
        return MODEL_LEVEL_LABELS.get(level, display_level(axis, level).capitalize())
    return display_level(axis, level).capitalize()


def _metric_label(metric: str) -> str:
    return metric.upper() if metric in _UPPERCASE_METRICS else metric.replace("_", " ").title()


# Lowest-effort publication-ish look: serif, no top/right border, faint gridlines, frameless legend.
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

# Fixed outcome colors, shared with plot_variance.py.
OUTCOME_COLORS = {
    "finished, valid": "tab:green",
    "extreme_outlier": "black",
    "finished, invalid": "tab:orange",
    "max_duration_reached": "tab:red",
    "missing_score": "tab:gray",
}


def outcome_category(df: pd.DataFrame) -> np.ndarray:
    """"finished, valid", "extreme_outlier", "finished, invalid", or the status itself."""
    finished = df["status"] == "finished"
    base = np.where(~finished, df["status"], np.where(df["valid"], "finished, valid", "finished, invalid"))
    return np.where(df["extreme_outlier"], "extreme_outlier", base)


def ordered_categories(df: pd.DataFrame) -> list[str]:
    """_outcome's distinct values in OUTCOME_COLORS order (so stacking is always
    green-then-orange-then-red, not alphabetical), any unanticipated status appended after."""
    present = set(df["_outcome"].unique())
    return [c for c in OUTCOME_COLORS if c in present] + sorted(present - OUTCOME_COLORS.keys())


def draw_outcome_score(ax_top: plt.Axes, ax_bot: plt.Axes, df: pd.DataFrame, axis: str, show_legend: bool, metric: str = "score") -> None:
    """Draw one task's two panels: outcome shares (all runs) above `metric` per level (valid runs).
    Use metric="gap_closed" for multi-task data.
    """
    levels = [lvl for lvl in AXIS_LEVELS[axis] if lvl in set(df[axis])]
    display_levels = [_level_label(axis, lvl) for lvl in levels]  # labels only -- grouping below stays on raw levels
    df = df.assign(_outcome=outcome_category(df))
    metric_label = _metric_label(metric_display_name(df, metric))

    categories = ordered_categories(df)
    counts_by_level = df.groupby(axis)["_outcome"].value_counts().unstack(fill_value=0)
    shares = counts_by_level.div(counts_by_level.sum(axis=1), axis=0).reindex(levels)
    bottom = np.zeros(len(levels))
    for cat in categories:
        heights = shares[cat].to_numpy()
        ax_top.bar(display_levels, heights, bottom=bottom, label=cat.replace("_", " ").capitalize(), color=OUTCOME_COLORS.get(cat, "tab:purple"))
        bottom += heights
    ax_top.set_ylabel("Share of runs")
    ax_top.set_ylim(0, 1)
    if show_legend:
        # Legend above the axes, wrapping instead of expanding.
        ax_top.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncols=min(len(categories), 3))

    valid = df[_metric_mask(df, metric)]  # extreme outliers excluded (own outcome-share color
    # above) so one collapsed run doesn't compress every other level's box onto a single visual scale
    per_level = [valid.loc[valid[axis] == lvl, metric].to_numpy() for lvl in levels]
    ax_bot.boxplot(per_level, positions=range(len(levels)), showfliers=False)
    rng = np.random.default_rng(0)  # fixed seed: jitter position shouldn't change between runs of this script
    for i, scores in enumerate(per_level):
        ax_bot.scatter(i + rng.uniform(-0.15, 0.15, size=len(scores)), scores, s=12, alpha=0.5, color="steelblue", edgecolors="black", linewidths=0.3)

    # gap_closed's anchors are fixed constants (0/1, no trivial); score's are this task's own
    # reference/backbone/trivial columns -- see utils.metric_anchor_lines for why the two differ.
    anchor_lines = metric_anchor_lines(df, metric)
    for value, style, label in anchor_lines:
        ax_bot.axhline(value, linestyle=style, color="black", linewidth=1, label=label)
    # Legend in the corner the anchor lines leave empty (lower right for gap_closed).
    if metric == "score":
        reference, backbone = df["reference"].dropna(), df["backbone"].dropna()
        corner = "upper right" if (len(backbone) and len(reference) and backbone.iloc[0] > reference.iloc[0]) else "lower right"
    else:
        corner = "lower right"
    ax_bot.legend(fontsize=8, loc=corner, frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
    ax_bot.set_xticks(range(len(levels)))
    ax_bot.set_xticklabels([f"{lvl}\n(n={len(s)})" for lvl, s in zip(display_levels, per_level)])
    ax_bot.set_ylabel(metric_label)
    ax_bot.set_xlabel(_axis_label(axis))


def plot_axis(df: pd.DataFrame, axis: str, out_path: Path, metric: str = "score") -> None:
    """One figure for df_task (a single task, or pooled tasks with metric="gap_closed")."""
    levels = [lvl for lvl in AXIS_LEVELS[axis] if lvl in set(df[axis])]
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(4, 1.6 * len(levels)) + 1.5, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]},
    )
    draw_outcome_score(ax_top, ax_bot, df, axis, show_legend=True, metric=metric)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_axis_faceted(df: pd.DataFrame, axis: str, out_path: Path, metric: str = "score") -> None:
    """One column per task; rows are never pooled across tasks. Outcome shares share a y-axis."""
    tasks = sorted(df["task"].unique())
    levels_per_task = [max(1, len({lvl for lvl in AXIS_LEVELS[axis] if lvl in set(df.loc[df["task"] == t, axis])})) for t in tasks]
    widths = [max(4, 1.6 * n) for n in levels_per_task]
    fig, axes = plt.subplots(
        2, len(tasks), figsize=(sum(widths) + 1.5, 8), squeeze=False, layout="constrained",
        gridspec_kw={"height_ratios": [1, 2], "width_ratios": widths},
    )
    for j, task in enumerate(tasks):
        draw_outcome_score(axes[0][j], axes[1][j], df[df["task"] == task], axis, show_legend=False, metric=metric)
        if j > 0:
            axes[0][j].sharey(axes[0][0])  # outcome share is a unitless proportion -- comparable across tasks

    # Center on the axes' actual bounding boxes, not the canvas.
    fig.canvas.draw()
    mid_x = (axes[0][0].get_position().x0 + axes[0][-1].get_position().x1) / 2
    top_y = axes[0][0].get_position().y1  # every column's top row shares this edge (constrained_layout aligns them)

    # One figure legend covering every task's categories, above the axes.
    categories = ordered_categories(df.assign(_outcome=outcome_category(df)))
    handles = [Patch(facecolor=OUTCOME_COLORS.get(c, "tab:purple"), label=c.replace("_", " ").capitalize()) for c in categories]
    # Column count scales with the number of task columns.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(mid_x, top_y + 0.02), ncols=min(len(categories), 3 * len(tasks)), fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _out_path(out_dir: Path, task: str, axis: str) -> Path:
    """<task first words>_<axis>.pdf."""
    task_label = "+".join(t.split("-")[0] for t in task.split("+"))
    return out_dir / "univariate" / f"{task_label}_{_axis_label(axis).lower()}.pdf"


def plot_grid(run_dir: Path, out_dir: Path | None = None, metric: str = "score") -> None:
    df = build_run_table([run_dir])
    task = df["task"].iloc[0]
    base = out_dir or REPO_ROOT / "analysis" / run_dir.name
    for axis in AXIS_NAMES:
        out_path = _out_path(base, task, axis)
        plot_axis(df, axis, out_path, metric)
        print(f"wrote {out_path}")


def plot_merged(run_dirs: list[Path], out_dir: Path | None = None, metric: str = "score") -> None:
    df = build_run_table(run_dirs)
    tasks = sorted(df["task"].unique())
    base = out_dir or REPO_ROOT / "analysis" / "+".join(d.name for d in run_dirs)
    for axis in AXIS_NAMES:
        out_path = _out_path(base, "+".join(tasks), axis)
        plot_axis_faceted(df, axis, out_path, metric)
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one or more grid run directories, e.g. runs/redshift-full-...")
    parser.add_argument(
        "--merge", action="store_true",
        help="one figure per axis with one column per task, instead of one set per grid dir",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write into this directory's univariate/ subfolder (set by analysis_single_task.py when chaining); pair with --merge for several run dirs",
    )
    parser.add_argument(
        "--metric", default="score", metavar="METRIC",
        help="column for the score row (default: score; gap_closed for multi-task data)",
    )
    args = parser.parse_args()
    if args.merge:
        plot_merged(args.run_dirs, args.out_dir, args.metric)
    else:
        for run_dir in args.run_dirs:
            plot_grid(run_dir, args.out_dir, args.metric)


if __name__ == "__main__":
    main()
