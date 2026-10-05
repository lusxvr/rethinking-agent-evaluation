"""Cross-task figures from analysis_cross_task.py's tables (default metric gap_closed).

Figures: per-axis level lines per task plus pooled (plot_pooled_axis_effects), a pooled bar chart
over gap_positive tasks (plot_pooled_axis_bars), an effect-size rank bump chart, a signed
lowest-to-highest-level heatmap, one axis's drilldown (--drilldown-axis) and per-task metric
distributions. Writes analysis/cross_<grid_label>/ unless --out-dir is given.

Usage: uv run python -m scripts.analysis.plot_cross_task runs/<grid1> [runs/<grid2> ...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from axes import AXIS_LEVELS, AXIS_NAMES
from scripts.analysis.analysis_cross_task import pooled_axis_table, rank_consistency_table, signed_ladder_effect_table
from scripts.analysis.utils import EFFECT_SIZE_EXCLUDE, _metric_mask, build_run_table, grid_label, metric_anchor_lines

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Shared figure style (copied per script, not imported).
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


def plot_pooled_axis_effects(
    df: pd.DataFrame, out_path: Path, min_valid: int = 3, metric: str = "gap_closed", qualifying_cells_only: bool = True,
) -> Path:
    """One panel per axis: level means per task plus a bold pooled line (qualifying-cells variant).
    Panels share the y range.
    """
    axes_list = list(AXIS_NAMES)
    tables = {axis: pooled_axis_table(df, axis, min_valid, metric, qualifying_cells_only) for axis in axes_list}
    tasks = sorted(df["task"].unique())
    value_cols = [*tasks, "pooled (equal-weight across tasks)"]
    all_vals = pd.concat([t[value_cols] for t in tables.values()]).to_numpy(dtype=float)
    pad = 0.05 * (np.nanmax(all_vals) - np.nanmin(all_vals) or 1.0)
    ylim = (np.nanmin(all_vals) - pad, np.nanmax(all_vals) + pad)

    widths = [max(2.2, 0.9 * len(AXIS_LEVELS[axis])) for axis in axes_list]
    fig, panel_axes = plt.subplots(
        1, len(axes_list), figsize=(sum(widths) + 1.5, 4.5), squeeze=False, layout="constrained",
        gridspec_kw={"width_ratios": widths},
    )

    handles = []
    for j, axis in enumerate(axes_list):
        ax = panel_axes[0][j]
        table = tables[axis]
        x = range(len(table))
        for k, task in enumerate(tasks):
            line, = ax.plot(x, table[task], marker="o", markersize=5, linewidth=1.6, color=f"C{k}", label=task)
            if j == 0:
                handles.append(line)
        pooled_line, = ax.plot(x, table["pooled (equal-weight across tasks)"], marker="D", markersize=6, linewidth=2.4, color="black", label="pooled (equal-weight)")
        if j == 0:
            handles.append(pooled_line)
        ax.axhline(0.0, color="gray", linewidth=0.7, linestyle=":")
        ax.set_ylim(ylim)
        ax.set_xlim(-0.4, len(table) - 0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(table.index, rotation=20, ha="right")
        ax.set_title(axis.capitalize())
        ax.tick_params(labelleft=(j == 0))
        if j == 0:
            ax.set_ylabel("Gap closed (0 = worse anchor, 1 = better anchor)" if metric == "gap_closed" else metric)

    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.1), frameon=False)
    # Short title; methodology belongs in the caption.
    fig.suptitle("Effect of each design axis on the capability gap closed", fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _gap_positive_tasks(df: pd.DataFrame) -> list[str]:
    """Tasks whose regime is gap_positive."""
    return sorted(df.loc[df["regime"] == "gap_positive", "task"].unique())


def plot_pooled_axis_bars(
    df: pd.DataFrame, out_path: Path, min_valid: int = 3, metric: str = "gap_closed", qualifying_cells_only: bool = True,
) -> Path:
    """Bar chart of pooled level means for every axis, with one dot per task. gap_positive tasks only,
    since gap_negative tasks measure a different question.
    """
    tasks = _gap_positive_tasks(df)
    df = df[df["task"].isin(tasks)]
    axes_list = list(AXIS_NAMES)
    tables = {axis: pooled_axis_table(df, axis, min_valid, metric, qualifying_cells_only) for axis in axes_list}

    # Groups laid out left to right with a visual gap between them -- own running x0, not a fixed
    # per-group width, since axes have different level counts (2-4).
    group_gap = 1.3
    positions: list[np.ndarray] = []
    level_labels: list[str] = []
    x0 = 0.0
    for axis in axes_list:
        n = len(tables[axis])
        xs = x0 + np.arange(n)
        positions.append(xs)
        level_labels.extend(tables[axis].index)
        x0 = xs[-1] + 1 + group_gap

    task_colors = {task: f"C{k}" for k, task in enumerate(tasks)}
    axis_colors = {axis: f"C{k}" for k, axis in enumerate(axes_list)}

    fig, ax = plt.subplots(figsize=(max(10.0, 0.9 * len(level_labels) + 2.0 * len(axes_list)), 5.5))
    for axis, xs in zip(axes_list, positions):
        table = tables[axis]
        ax.bar(xs, table["pooled (equal-weight across tasks)"], width=0.7, color=axis_colors[axis], alpha=0.75, edgecolor="black", linewidth=0.6, zorder=2)
        for task in tasks:
            ax.scatter(xs, table[task], s=28, color=task_colors[task], edgecolors="black", linewidths=0.4, zorder=3)

    for xs_prev, xs_next in zip(positions, positions[1:]):
        ax.axvline((xs_prev[-1] + xs_next[0]) / 2, color="lightgray", linewidth=0.8, zorder=0)
    for value, style, label in metric_anchor_lines(df, metric):
        ax.axhline(value, linestyle=style, color="gray", linewidth=1, zorder=1)

    ax.set_xlim(positions[0][0] - 0.8, positions[-1][-1] + 0.8)
    ax.set_xticks(np.concatenate(positions))
    ax.set_xticklabels(level_labels, rotation=30, ha="right")
    for axis, xs in zip(axes_list, positions):
        ax.text(xs.mean(), 1.02, axis.capitalize(), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=11, fontweight="bold")

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=task_colors[t], markeredgecolor="black", label=t) for t in tasks]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=len(tasks), frameon=False, fontsize=9)

    ax.set_ylabel("Gap closed (0 = backbone, 1 = specialist recipe)" if metric == "gap_closed" else metric)
    # pad clears the per-group axis-name labels (drawn at axes-fraction y=1.02, right at the top
    # edge) -- the default title pad sits close enough to that row to overlap it otherwise.
    ax.set_title("Effect of each design axis on the capability gap closed (gap-positive tasks)", fontsize=13, pad=28)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rank_consistency(
    df: pd.DataFrame, out_path: Path, min_valid: int = 3, metric: str = "gap_closed", exclude: dict[str, tuple[str, ...]] | None = None,
) -> Path:
    """Bump chart of each axis's effect-size rank per task plus the mean, labeled directly at the line ends."""
    rank_table, _ = rank_consistency_table(df, min_valid, metric, exclude)
    tasks = sorted(df["task"].unique())
    n_axes = len(rank_table)
    xs = list(range(len(tasks)))
    mean_x = len(tasks) + 0.6  # separated gap before the summary column

    fig, ax = plt.subplots(figsize=(2.2 * len(tasks) + 3.5, 0.65 * n_axes + 1.3))
    for i, axis in enumerate(rank_table.index):
        color = f"C{i}"
        y = rank_table.loc[axis, tasks].to_numpy(dtype=float)
        ax.plot(xs, y, marker="o", markersize=7, linewidth=2, color=color)
        ax.plot([xs[-1], mean_x], [y[-1], rank_table.loc[axis, "mean rank"]], marker="o", markersize=7, linewidth=2, color=color, linestyle=":")
        ax.text(xs[0] - 0.15, y[0], axis, ha="right", va="center", fontsize=9, color=color, fontweight="bold")

    ax.axvline((xs[-1] + mean_x) / 2, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(xs[0] - 1.4, mean_x + 0.4)
    ax.set_xticks([*xs, mean_x])
    ax.set_xticklabels([*tasks, "mean"], rotation=15, ha="right")
    ax.set_yticks(range(1, n_axes + 1))
    ax.invert_yaxis()
    ax.set_ylabel("Rank (1 = biggest effect)")
    ax.set_title("Which design axis matters most, and how consistently" + (" (information.protocol excluded)" if exclude else ""), fontsize=13)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_headline_heatmap(df: pd.DataFrame, out_path: Path, min_valid: int = 3, metric: str = "gap_closed") -> Path:
    """signed_ladder_effect_table as a diverging heatmap centered at 0, with values printed in cells."""
    table = signed_ladder_effect_table(df, min_valid, metric)
    tasks = list(table.columns)
    values = table.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) or 1.0

    fig, ax = plt.subplots(figsize=(1.7 * len(tasks) + 3, 0.75 * len(table) + 1.5))
    ax.grid(False)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([axis.capitalize() for axis in table.index])
    for i in range(len(table)):
        for j in range(len(tasks)):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=11,
                         color="white" if abs(v) > 0.6 * vmax else "black")
    ax.set_xticks(np.arange(len(tasks) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(table) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(f"{metric}: highest ladder level minus lowest")
    ax.set_title(f"Which design factors move {metric}, and in which direction")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_axis_drilldown(
    df: pd.DataFrame, out_path: Path, axis: str, min_valid: int = 3, metric: str = "gap_closed", qualifying_cells_only: bool = True,
) -> Path:
    """One axis's pooled_axis_table as a standalone line figure."""
    table = pooled_axis_table(df, axis, min_valid, metric, qualifying_cells_only)
    tasks = sorted(df["task"].unique())
    x = list(range(len(table)))

    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(table)) + 2, 5.5))
    for k, task in enumerate(tasks):
        ax.plot(x, table[task], marker="o", markersize=7, linewidth=2, color=f"C{k}")
        ax.text(x[-1] + 0.08, table[task].iloc[-1], task, va="center", fontsize=10, color=f"C{k}")
    ax.plot(x, table["pooled (equal-weight across tasks)"], marker="D", markersize=8, linewidth=3, color="black")
    ax.text(x[-1] + 0.08, table["pooled (equal-weight across tasks)"].iloc[-1], "pooled (equal-weight)", va="center", fontsize=10, color="black", fontweight="bold")

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(-0.4, len(table) - 1 + 1.9)  # headroom on the right for the end-labels
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylabel(metric)
    ax.set_title(f"{axis.capitalize()}'s effect on {metric}, by task")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_capability_profile(df: pd.DataFrame, out_path: Path, metric: str = "gap_closed") -> Path:
    """Box and strip of every valid run's `metric` per task, across the whole grid."""
    tasks = sorted(df["task"].unique())
    data = [df.loc[_metric_mask(df, metric) & (df["task"] == t), metric].to_numpy() for t in tasks]

    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(tasks)) + 1.5, 5))
    ax.boxplot(data, positions=range(len(tasks)), showfliers=False)
    rng = np.random.default_rng(0)  # fixed seed: jitter position shouldn't change between runs
    for i, vals in enumerate(data):
        ax.scatter(i + rng.uniform(-0.15, 0.15, size=len(vals)), vals, s=14, alpha=0.5, color="steelblue", edgecolors="black", linewidths=0.3)
    if metric == "gap_closed":
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{t}\n(n={len(v)})" for t, v in zip(tasks, data)], rotation=15, ha="right")
    ax.set_ylabel("Gap closed (0 = worse anchor, 1 = better anchor)" if metric == "gap_closed" else metric)
    ax.set_title("Capability profile: how much of the achievable gap agents close, by task")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one or more grid run directories, spanning any number of tasks")
    parser.add_argument("--min-valid", type=int, default=3, help="min valid replicates/cell for the noise-floor std and the qualifying-cells ablation (default 3)")
    parser.add_argument("--metric", default="gap_closed", metavar="METRIC", help="a numeric column of build_run_table (default: gap_closed -- see module docstring)")
    parser.add_argument("--drilldown-axis", default="information", choices=AXIS_NAMES, help="which axis plot_axis_drilldown zooms into (default: information -- see module docstring)")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write here instead of analysis/cross_<grid_label>/ (set by analysis_cross_task.py)",
    )
    args = parser.parse_args()

    df = build_run_table(args.run_dirs)

    # Named by the task set, so different subsets do not overwrite each other.
    out_dir = args.out_dir or REPO_ROOT / "analysis" / f"cross_{grid_label(df)}"
    for path in (
        plot_pooled_axis_effects(df, out_dir / f"pooled_axis_effect_{args.metric}.pdf", args.min_valid, args.metric),
        plot_pooled_axis_bars(df, out_dir / f"pooled_axis_bars_{args.metric}.pdf", args.min_valid, args.metric),
        plot_rank_consistency(df, out_dir / f"rank_consistency_{args.metric}.pdf", args.min_valid, args.metric),
        plot_rank_consistency(df, out_dir / f"rank_consistency_{args.metric}_protocol_excluded.pdf", args.min_valid, args.metric, EFFECT_SIZE_EXCLUDE),
        plot_headline_heatmap(df, out_dir / f"headline_heatmap_{args.metric}.pdf", args.min_valid, args.metric),
        plot_axis_drilldown(df, out_dir / f"headline_drilldown_{args.drilldown_axis}_{args.metric}.pdf", args.drilldown_axis, args.min_valid, args.metric),
        plot_capability_profile(df, out_dir / f"capability_profile_{args.metric}.pdf", args.metric),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
