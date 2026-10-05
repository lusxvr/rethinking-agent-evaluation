"""Two-axis interactions: bubble matrix of mean metric (color) and n (size) for pairs of the top-k
axes by effect size, per task. All pairs of one call share color and size scales.

Writes analysis/<grid>/interactions/<task>_<axis_a>-<axis_b>.pdf (or under --out-dir). --merge
pools a task's grid dirs (e.g. one per model tier); tasks are never pooled.

Usage: uv run python -m scripts.analysis.plot_interactions runs/<grid> [...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from axes import AXIS_LEVELS, AXIS_NAMES, display_level
from scripts.analysis.utils import EFFECT_SIZE_EXCLUDE, _axis_effect_sizes, _metric_mask, _section, build_run_table, metric_display_name

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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
    "axes.grid": False,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})


def top_axes(df_task: pd.DataFrame, k: int, min_valid: int = 3, metric: str = "score") -> list[str]:
    """The k axes print_effect_sizes ranks highest (information.protocol excluded, as in the text report)."""
    table = _axis_effect_sizes(df_task, min_valid, EFFECT_SIZE_EXCLUDE, metric)
    return table.sort_values("rank").index[:k].tolist()


def pair_matrices(df_task: pd.DataFrame, axis_a: str, axis_b: str, metric: str = "score") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marginal mean `metric` and n, one cell per (axis_a level, axis_b level), every other axis
    pooled out -- same marginalization as analysis_single_task's axis tables, just 2D instead of 1D."""
    valid = df_task[_metric_mask(df_task, metric)]  # extreme outliers excluded -- see utils._metric_mask
    levels_a = [lvl for lvl in AXIS_LEVELS[axis_a] if lvl in set(valid[axis_a])]
    levels_b = [lvl for lvl in AXIS_LEVELS[axis_b] if lvl in set(valid[axis_b])]
    grouped = valid.groupby([axis_a, axis_b])[metric]
    means = grouped.mean().unstack(axis_b).reindex(index=levels_a, columns=levels_b)
    counts = grouped.count().unstack(axis_b).reindex(index=levels_a, columns=levels_b)
    return means, counts


# Marker area range (points^2); area scales linearly with n.
MARKER_AREA_RANGE = (500, 3200)

# Metric name as a label: known abbreviations upper-cased, others title-cased.
_UPPERCASE_METRICS = {"r2", "mcc"}


def _metric_label(metric: str) -> str:
    return metric.upper() if metric in _UPPERCASE_METRICS else metric.replace("_", " ").title()


def _shared_bounds(df_task: pd.DataFrame, pairs: list[tuple[str, str]], metric: str) -> tuple[float, float, float, float]:
    """Color (mean metric) and size (n) bounds across all pairs, so panels are comparable."""
    all_means, all_counts = [], []
    for axis_a, axis_b in pairs:
        means, counts = pair_matrices(df_task, axis_a, axis_b, metric)
        all_means.append(means.to_numpy(dtype=float))
        all_counts.append(counts.to_numpy(dtype=float))
    means_stack = np.concatenate([m.ravel() for m in all_means])
    counts_stack = np.concatenate([c.ravel() for c in all_counts])
    counts_stack = counts_stack[~np.isnan(counts_stack)]
    return float(np.nanmin(means_stack)), float(np.nanmax(means_stack)), float(counts_stack.min()), float(counts_stack.max())


def plot_pair(
    df_task: pd.DataFrame, axis_a: str, axis_b: str, out_path: Path, metric: str = "score",
    vmin: float | None = None, vmax: float | None = None, n_lo: float | None = None, n_hi: float | None = None,
) -> None:
    """Bubble matrix for one axis pair: color = mean `metric`, size = n. Without bounds, scales to
    this pair alone.
    """
    metric_label = _metric_label(metric_display_name(df_task, metric))
    means, counts = pair_matrices(df_task, axis_a, axis_b, metric)
    y_axis, x_axis = axis_a, axis_b
    if x_axis == "budget":  # budget always plotted on the y-axis, whichever axis ranked as axis_a
        means, counts = means.T, counts.T
        y_axis, x_axis = x_axis, y_axis
    n_rows, n_cols = len(means.index), len(means.columns)

    # Long-form (row, col, mean, n) per non-missing cell -- scatter takes flat coordinate/size/color
    # arrays, not a 2D grid the way imshow did.
    rows_idx, cols_idx = np.meshgrid(range(n_rows), range(n_cols), indexing="ij")
    means_arr, counts_arr = means.to_numpy(dtype=float), counts.to_numpy(dtype=float)
    have_data = ~np.isnan(means_arr)
    ys, xs, vals, ns = rows_idx[have_data], cols_idx[have_data], means_arr[have_data], counts_arr[have_data]

    # Size per cell plus fixed margins for labels and colorbar.
    fig, ax = plt.subplots(figsize=(1.3 * n_cols + 1.6, 1.1 * n_rows + 1.0))

    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin if vmin is not None else np.nanmin(means_arr), vmax if vmax is not None else np.nanmax(means_arr))
    n_lo = n_lo if n_lo is not None else ns.min()
    n_hi = n_hi if n_hi is not None else ns.max()
    area_lo, area_hi = MARKER_AREA_RANGE

    def area_for_n(n):
        return (area_lo + area_hi) / 2 if n_hi == n_lo else area_lo + (area_hi - area_lo) * (n - n_lo) / (n_hi - n_lo)

    scat = ax.scatter(xs, ys, s=area_for_n(ns), c=vals, cmap=cmap, norm=norm, edgecolors="black", linewidths=0.6)
    cbar = fig.colorbar(scat, ax=ax, shrink=0.8, label=metric_label)

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(n_rows - 0.4, -0.6)  # inverted -- row 0 (first level) at the top, matching the old imshow layout
    ax.set_xticks(range(n_cols), [_level_label(x_axis, c) for c in means.columns])
    ax.set_yticks(range(n_rows), [_level_label(y_axis, r) for r in means.index])
    ax.set_xlabel(_axis_label(x_axis))
    ax.set_ylabel(_axis_label(y_axis))
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    # Size note anchored to the colorbar, so its position does not depend on the column count.
    cbar_box = cbar.ax.get_position()
    fig.text(cbar_box.x0, cbar_box.y0 - 0.03, "Marker size ∝\nvalid runs",
              ha="left", va="top", fontsize=8, color="gray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _relabeled(df: pd.DataFrame, axis_a: str, axis_b: str) -> pd.DataFrame:
    """Rename both axes' levels with display_level, as in the plots."""
    return df.rename(index=lambda r: display_level(axis_a, r), columns=lambda c: display_level(axis_b, c))


def print_pair(df_task: pd.DataFrame, axis_a: str, axis_b: str, metric: str = "score") -> None:
    means, counts = pair_matrices(df_task, axis_a, axis_b, metric)
    _section(f"{axis_a} x {axis_b} ({'+'.join(sorted(df_task['task'].unique()))})")
    print("mean:")
    print(_relabeled(means, axis_a, axis_b).to_string(float_format=lambda x: f"{x:.4f}"))
    print("n:")
    print(_relabeled(counts, axis_a, axis_b).to_string())


def _out_path(out_dir: Path, task: str, axis_a: str, axis_b: str) -> Path:
    """<task first words>_<axis_a>-<axis_b>.pdf."""
    task_label = "+".join(t.split("-")[0] for t in task.split("+"))
    a_label, b_label = _axis_label(axis_a).lower(), _axis_label(axis_b).lower()
    return out_dir / "interactions" / f"{task_label}_{a_label}-{b_label}.pdf"


def plot_interactions(df_task: pd.DataFrame, out_dir: Path, k: int = 2, min_valid: int = 3, metric: str = "score") -> None:
    """All top-axis pairs for df_task (one task, or pooled tasks with metric="gap_closed")."""
    task = "+".join(sorted(df_task["task"].unique()))
    top = top_axes(df_task, k, min_valid, metric)
    others = [a for a in AXIS_NAMES if a != "model"]
    pairs = []
    seen = set()
    for axis_a in top:
        for axis_b in others:
            pair = frozenset((axis_a, axis_b))
            if axis_b == axis_a or pair in seen:
                continue
            seen.add(pair)
            pairs.append((axis_a, axis_b))

    # Shared color and size scales across this call's pairs.
    vmin, vmax, n_lo, n_hi = _shared_bounds(df_task, pairs, metric) if pairs else (0.0, 1.0, 1.0, 1.0)
    for axis_a, axis_b in pairs:
        print_pair(df_task, axis_a, axis_b, metric)
        plot_pair(df_task, axis_a, axis_b, _out_path(out_dir, task, axis_a, axis_b), metric, vmin, vmax, n_lo, n_hi)


def budget_bar_table(df: pd.DataFrame, axis: str, metric: str = "gap_closed") -> pd.DataFrame:
    """Mean, 95% CI half-width (1.96 SE) and n of `metric` per (axis level, budget), over usable runs."""
    valid = df[_metric_mask(df, metric)]
    g = valid.groupby([axis, "budget"], observed=True)[metric]
    table = pd.DataFrame({"mean": g.mean(), "n": g.count()})
    table["ci95"] = 1.96 * g.std() / np.sqrt(table["n"])
    return table


def plot_budget_bars(df: pd.DataFrame, axis: str, out_path: Path, metric: str = "gap_closed", width_in: float | None = None) -> Path:
    """Paper's interaction figure: mean `metric` with 95% CI (left) and usable runs (right) per level
    of `axis`, one bar per budget level. width_in overrides the figure width."""
    table = budget_bar_table(df, axis, metric)
    levels = [lvl for lvl in AXIS_LEVELS.get(axis, ()) if lvl in table.index.get_level_values(0)]
    budgets = [b for b in AXIS_LEVELS["budget"] if b in table.index.get_level_values(1)]
    pivot = {col: table[col].unstack("budget").reindex(index=levels, columns=budgets) for col in ("mean", "ci95", "n")}

    colors = [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 0.9, len(budgets))]
    x = np.arange(len(levels))
    width = 0.26
    offsets = (np.arange(len(budgets)) - (len(budgets) - 1) / 2) * width
    pane_width = max(4, 1.6 * len(levels)) + 1.0
    with plt.rc_context({"font.size": 14, "mathtext.fontset": "cm", "axes.linewidth": 0.8}):
        fig, axes = plt.subplots(1, 2, figsize=(width_in or 2 * pane_width, 4.4))
        for ax, values, errors, ylabel in ((axes[0], pivot["mean"], pivot["ci95"], r"$\mathcal{G}$" if metric == "gap_closed" else metric_display_name(df, metric)),
                                           (axes[1], pivot["n"], None, "Completed Runs")):
            for j, budget in enumerate(budgets):
                ax.bar(x + offsets[j], values[budget].to_numpy(), width=width, color=colors[j], edgecolor="black",
                       linewidth=0.5, zorder=3, yerr=None if errors is None else errors[budget].to_numpy(), capsize=2,
                       error_kw={"lw": 0.8, "capthick": 0.8, "ecolor": "black", "zorder": 4}, label=budget.capitalize())
            ax.set_ylabel(ylabel)
            ax.set_xlabel(_axis_label(axis))
            ax.set_xticks(x)
            ax.set_xticklabels([_level_label(axis, lvl) for lvl in levels])
            ax.set_xlim(-0.55, len(levels) - 0.45)
            ax.yaxis.grid(True, color="0.9", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
        axes[1].set_ylim(0, np.nanmax(pivot["n"].to_numpy()) * 1.05)
        lo = np.floor(np.nanmin((pivot["mean"] - pivot["ci95"]).to_numpy()) * 10) / 10
        hi = np.ceil(np.nanmax((pivot["mean"] + pivot["ci95"]).to_numpy()) * 10) / 10
        axes[0].set_yticks(np.arange(lo, hi + 1e-9, 0.1))
        axes[0].set_ylim(lo, hi + 0.02)
        axes[0].legend(title="Budget", loc="upper left", borderaxespad=0.1, borderpad=0.3, labelspacing=0.25, frameon=True,
                       facecolor="white", framealpha=0.9, edgecolor="none", handlelength=1.1, handletextpad=0.5)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_grid(run_dir: Path, k: int = 2, min_valid: int = 3, out_dir: Path | None = None, metric: str = "score") -> None:
    df = build_run_table([run_dir])
    base = out_dir or REPO_ROOT / "analysis" / run_dir.name
    for task in sorted(df["task"].unique()):
        plot_interactions(df[df["task"] == task], base, k, min_valid, metric)


def plot_merged(run_dirs: list[Path], k: int = 2, min_valid: int = 3, out_dir: Path | None = None, metric: str = "score") -> None:
    """Pools rows within a task across run dirs before ranking/plotting -- see module docstring
    for why this differs from plot_univariate/plot_variance's --merge (which facets, never pools)."""
    df = build_run_table(run_dirs)
    base = out_dir or REPO_ROOT / "analysis" / "+".join(d.name for d in run_dirs)
    for task in sorted(df["task"].unique()):
        plot_interactions(df[df["task"] == task], base, k, min_valid, metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one or more grid run directories, e.g. runs/redshift-full-...")
    parser.add_argument("-k", type=int, default=2, help="how many top-ranked axes to scope interactions to (default 2)")
    parser.add_argument("--min-valid", type=int, default=3, help="min valid replicates/cell for the ranking that picks the top axes (default 3)")
    parser.add_argument(
        "--merge", action="store_true",
        help="pool a task's run dirs before plotting, e.g. grids split by model tier",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write into this directory's interactions/ subfolder (set by analysis_single_task.py when chaining)",
    )
    parser.add_argument(
        "--metric", default="score", metavar="METRIC",
        help="a numeric run-table column (default: score; gap_closed for multi-task data)",
    )
    args = parser.parse_args()

    if args.merge:
        plot_merged(args.run_dirs, args.k, args.min_valid, args.out_dir, args.metric)
    else:
        for run_dir in args.run_dirs:
            plot_grid(run_dir, args.k, args.min_valid, args.out_dir, args.metric)


if __name__ == "__main__":
    main()
