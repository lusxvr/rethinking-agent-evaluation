"""Replicate-variance plots matching analysis_single_task's variance sections.

The CLI writes analysis/<grid>/variance/<task>_overview.pdf with four panels (completion and
hurdle, score variance, CI, noise floor). The paper's figures come from plot_variance_main and
plot_variance_appendix via analysis_cross_task.py. See docs/analysis-metrics.md §10.

Usage: uv run python -m scripts.analysis.plot_variance runs/<grid> [...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.analysis.analysis_single_task import _ci_stability_table
from scripts.analysis.utils import (
    CELL_AXES,
    _SUCCESS_INDICATOR,
    _conditional_noise_floor_metrics,
    _metric_mask,
    _noise_floor_metrics,
    build_run_table,
    metric_anchor_lines,
    metric_display_name,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Shared figure style (copied per script, not imported).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})

# Manual overrides for _draw_score_variance_panel's legend, where the backbone-vs-reference rule
# of thumb picks a corner the actual data disagrees with -- see that function's comment.
SCORE_LEGEND_CORNER_OVERRIDES = {"redshift-estimation": "upper right"}

# Hurdle outcome colors shared by the completion and score panels: red never, orange mixed,
# green always clears the hurdle.
_OUTCOME_COLORS = {"never": "#e57373", "mixed": "#ffb74d", "always": "#81c784"}
_NO_COMPLETION_COLOR = "#bdbdbd"

# Accent colors without a good/bad meaning, matching the interaction figures' blues.
_BLUES = plt.get_cmap("Blues")
_CI_ALL_COLOR = _BLUES(0.35)
_CI_CONDITIONAL_COLOR = _BLUES(0.9)
_NOISE_SIGNAL_COLOR = _BLUES(0.625)

# Marker per task when pooled (sorted for stability); color carries the hurdle outcome.
_TASK_MARKER_CYCLE = ("o", "s", "^", "D", "P", "X")

# Metric names shown upper-case.
_UPPERCASE_METRICS = {"r2", "mcc"}


def _metric_label(metric: str) -> str:
    return metric.upper() if metric in _UPPERCASE_METRICS else metric.replace("_", " ").title()


def _draw_completion_panel(ax_complete: plt.Axes, df: pd.DataFrame) -> None:
    """Cells by number of completed replicates, stacked by how many clear the hurdle (all/some/none)."""
    cells = df.groupby(CELL_AXES)
    n_reps = cells.size()
    n_valid = cells["valid"].apply(lambda s: (s == True).sum())  # noqa: E712
    cleared = (df["valid"] == True) & (df[_SUCCESS_INDICATOR] == True)  # noqa: E712
    n_cleared = cleared.groupby([df[axis] for axis in CELL_AXES]).sum()

    max_reps = int(n_reps.max())
    x_range = range(max_reps + 1)
    outcome = pd.Series("none_completed", index=n_valid.index)
    outcome[(n_valid > 0) & (n_cleared == n_valid)] = "always"
    outcome[(n_valid > 0) & (n_cleared > 0) & (n_cleared < n_valid)] = "mixed"
    outcome[(n_valid > 0) & (n_cleared == 0)] = "never"
    counts = pd.crosstab(n_valid, outcome).reindex(index=x_range, fill_value=0)
    bottom = pd.Series(0, index=x_range, dtype=int)
    stack = (
        ("none_completed", _NO_COMPLETION_COLOR, "No Completed Run"),
        ("never", _OUTCOME_COLORS["never"], "None Clear Hurdle"),
        ("mixed", _OUTCOME_COLORS["mixed"], "Some Clear Hurdle"),
        ("always", _OUTCOME_COLORS["always"], "All Clear Hurdle"),
    )
    for key, color, _ in stack:
        values = counts[key] if key in counts else pd.Series(0, index=x_range)
        ax_complete.bar(x_range, values, bottom=bottom, color=color, edgecolor="white")
        bottom = bottom + values
    ax_complete.set_xlabel(f"Completed Runs per Cell (of {max_reps})")
    ax_complete.set_ylabel("Cells")
    ax_complete.set_xticks(range(max_reps + 1))
    ax_complete.legend(
        handles=[Patch(color=color, label=label) for _, color, label in stack],
        fontsize=10, loc="upper left", frameon=True, facecolor="white", framealpha=0.85, edgecolor="none",
    )


def _draw_score_variance_panel(
    ax_score: plt.Axes, df: pd.DataFrame, min_valid: int, metric: str = "score",
    score_legend_corner: str | None = None, show_trend: bool = True,
) -> None:
    """Each qualifying cell's mean (x) vs std (y), colored by its hurdle success rate, shaped by task
    when pooled. show_trend adds an OLS line and the Spearman correlation.
    """
    metric_label = _metric_label(metric_display_name(df, metric))
    anchor_lines = metric_anchor_lines(df, metric)

    valid = df[_metric_mask(df, metric)]  # extreme outliers excluded -- see utils._metric_mask
    counts = valid.groupby(CELL_AXES)[metric].count()
    qualifying = counts[counts >= min_valid].index
    cell_mean = valid.groupby(CELL_AXES)[metric].mean().loc[qualifying] if len(qualifying) else pd.Series(dtype=float)
    cell_std = valid.groupby(CELL_AXES)[metric].std().loc[qualifying] if len(qualifying) else pd.Series(dtype=float)
    cell_p_hat = (
        valid.assign(**{_SUCCESS_INDICATOR: valid[_SUCCESS_INDICATOR].astype(float)}).groupby(CELL_AXES)[_SUCCESS_INDICATOR].mean().loc[qualifying]
        if len(qualifying) else pd.Series(dtype=float)
    )

    tasks_present = sorted(df["task"].unique())
    task_marker = dict(zip(tasks_present, _TASK_MARKER_CYCLE))
    if len(qualifying):
        point_colors = cell_p_hat.map(lambda p: _OUTCOME_COLORS["always"] if p >= 1.0 else _OUTCOME_COLORS["never"] if p <= 0.0 else _OUTCOME_COLORS["mixed"])
        cell_tasks = cell_mean.index.get_level_values("task")
        for task in tasks_present:
            mask = cell_tasks == task
            ax_score.scatter(
                cell_mean[mask], cell_std[mask], s=20, alpha=0.75, c=point_colors[mask].to_numpy(),
                marker=task_marker[task], edgecolors="black", linewidths=0.3,
            )
    ax_score.set_xlabel(r"Mean $\mathcal{G}$ per Cell" if metric == "gap_closed" else f"Mean {metric_label} per Cell")
    ax_score.set_ylabel("Standard Deviation")
    # ax_score.set_title(f"{metric_label} variance (≥{min_valid} valid replicates)")

    if show_trend and len(qualifying) >= 2:
        slope, intercept = np.polyfit(cell_mean, cell_std, 1)
        xs = np.array([cell_mean.min(), cell_mean.max()])
        ax_score.plot(xs, slope * xs + intercept, color="black", linewidth=1.5, zorder=5)
        rho = cell_mean.corr(cell_std, method="spearman")
        ax_score.text(
            0.03, 0.05, f"Spearman ρ = {rho:.2f}", transform=ax_score.transAxes,
            ha="left", va="bottom", fontsize=14, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

    # gap_closed's anchors are fixed at 0/1, so no anchor lines; a task-shape legend if pooled.
    if metric == "score":
        for value, style, label in anchor_lines:
            ax_score.axvline(value, linestyle=style, color="black", linewidth=1, label=label)
        if anchor_lines:
            # Corner the anchors leave empty; score_legend_corner overrides it per task.
            reference = df["reference"].dropna().iloc[0] if df["reference"].notna().any() else None
            backbone = df["backbone"].dropna().iloc[0] if df["backbone"].notna().any() else None
            default_corner = "upper right" if (backbone is not None and reference is not None and backbone > reference) else "lower right"
            corner = score_legend_corner or default_corner
            ax_score.legend(fontsize=14, loc=corner, frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
    elif len(tasks_present) > 1:
        ax_score.legend(
            handles=[
                Line2D([0], [0], marker=task_marker[task], color="none", markerfacecolor="lightgray",
                       markeredgecolor="black", markersize=7, label=task.split("-")[0])
                for task in tasks_present
            ],
            fontsize=14, loc="upper right", frameon=True, facecolor="white", framealpha=0.85, edgecolor="none",
        )


def _draw_ci_panel(ax_ci: plt.Axes, df: pd.DataFrame, sample_sizes: tuple[int, ...], metric: str = "score") -> None:
    """Mean 95% CI half-width at 3, 4 and 5 replicates, for all runs and for beats-trivial runs only."""
    ci_table = _ci_stability_table(df, sample_sizes, metric)
    ci_table_conditional = _ci_stability_table(df[df[_SUCCESS_INDICATOR] == True], sample_sizes, metric)  # noqa: E712
    x = range(len(ci_table.index))
    width = 0.38
    bars_all = ax_ci.bar([i - width / 2 for i in x], ci_table["mean 95% CI half-width"], width, color=_CI_ALL_COLOR, edgecolor="white", label="All Completed Runs")
    bars_cond = ax_ci.bar(
        [i + width / 2 for i in x], ci_table_conditional["mean 95% CI half-width"], width,
        color=_CI_CONDITIONAL_COLOR, edgecolor="white", label="Clears Hurdle",
    )
    ax_ci.bar_label(bars_all, labels=[f"n={n}" for n in ci_table["cells"]], fontsize=9, padding=2)
    ax_ci.bar_label(bars_cond, labels=[f"n={n}" for n in ci_table_conditional["cells"]], fontsize=9, padding=2)
    ax_ci.set_xticks(list(x), [str(k) for k in ci_table.index])
    ax_ci.set_xlabel("Runs per Cell")
    ax_ci.set_ylabel("Mean 95% CI Half-Width")
    ax_ci.legend(fontsize=10, loc="upper right", frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")


def _draw_noise_floor_panel(ax_noise: plt.Axes, df: pd.DataFrame, min_valid: int, metric: str = "score") -> None:
    """Within- vs between-cell variance share, for all runs and for beats-trivial runs only."""

    def _variance_shares(metrics: dict) -> tuple[float, float] | None:
        within_std, between_std = metrics.get("within-cell std (noise floor)"), metrics.get("between-cell std (bias-corrected)")
        if within_std is None or between_std is None:
            return None
        within_var, between_var = within_std**2, between_std**2
        total = within_var + between_var
        icc = between_var / total if total else float("nan")
        return (icc, 1 - icc) if total else (0.0, 0.0)

    bar_shares = {
        "All Completed Runs": _variance_shares(_noise_floor_metrics(df, min_valid, metric)),
        "Clears Hurdle": _variance_shares(_conditional_noise_floor_metrics(df, min_valid, metric)),
    }
    if all(shares is None for shares in bar_shares.values()):
        ax_noise.axis("off")
        ax_noise.text(0.5, 0.5, "not enough qualifying cells", ha="center", va="center")
    else:
        xs, labels = [], []
        for i, (label, shares) in enumerate(bar_shares.items()):
            if shares is None:
                continue
            between_share, within_share = shares
            # Neutral colors: a variance share, not a pass/fail outcome.
            ax_noise.bar([i], [between_share], color=_NOISE_SIGNAL_COLOR, edgecolor="white", label="Between-Cell (Signal)" if i == 0 else None)
            ax_noise.bar([i], [within_share], bottom=[between_share], color="lightgray", edgecolor="white", label="Within-Cell (Noise)" if i == 0 else None)
            if between_share == between_share:  # not NaN
                ax_noise.text(i, 1.05, f"ICC={between_share:.1%}", ha="center", fontsize=13)
            xs.append(i)
            labels.append(label)
        ax_noise.set_ylim(0, 1.1)
        ax_noise.xaxis.grid(False)  # x is just two category labels, not a numeric axis -- no vertical gridline to draw
        ax_noise.set_xticks(xs, labels, fontsize=14)
        ax_noise.set_ylabel("Share of Total Variance")
        ax_noise.legend(fontsize=14, loc="lower center", frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
    # ax_noise.set_title("Noise floor: all runs vs. beats trivial")


def draw_variance_panels(
    ax_complete: plt.Axes, ax_score: plt.Axes, ax_ci: plt.Axes, ax_noise: plt.Axes,
    df: pd.DataFrame, min_valid: int, sample_sizes: tuple[int, ...],
    score_legend_corner: str | None = None, metric: str = "score",
) -> None:
    """All four panels for one task (the CLI's diagnostic figure)."""
    _draw_completion_panel(ax_complete, df)
    _draw_score_variance_panel(ax_score, df, min_valid, metric, score_legend_corner)
    _draw_ci_panel(ax_ci, df, sample_sizes, metric)
    _draw_noise_floor_panel(ax_noise, df, min_valid, metric)


def plot_variance_overview(
    df: pd.DataFrame, out_path: Path, min_valid: int = 3, sample_sizes: tuple[int, ...] = (3, 4, 5), metric: str = "score",
) -> None:
    """One four-panel figure for df_task."""
    task = df["task"].iloc[0]
    fig, (ax_complete, ax_score, ax_ci, ax_noise) = plt.subplots(1, 4, figsize=(21, 4.8))
    draw_variance_panels(ax_complete, ax_score, ax_ci, ax_noise, df, min_valid, sample_sizes, SCORE_LEGEND_CORNER_OVERRIDES.get(task), metric)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance_faceted(
    df: pd.DataFrame, out_path: Path, min_valid: int = 3, sample_sizes: tuple[int, ...] = (3, 4, 5), metric: str = "score",
) -> None:
    """One row per task; rows are never pooled across tasks."""
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(len(tasks), 4, figsize=(21, 4.8 * len(tasks)), squeeze=False, layout="constrained")
    for i, task in enumerate(tasks):
        draw_variance_panels(
            axes[i][0], axes[i][1], axes[i][2], axes[i][3], df[df["task"] == task], min_valid, sample_sizes,
            SCORE_LEGEND_CORNER_OVERRIDES.get(task), metric,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance_main(df: pd.DataFrame, out_path: Path, min_valid: int = 3, metric: str = "score") -> None:
    """Paper main-text figure: score variance with trend, and the noise floor."""
    task = df["task"].iloc[0]
    fig, (ax_score, ax_noise) = plt.subplots(1, 2, figsize=(11, 4.8))
    _draw_score_variance_panel(ax_score, df, min_valid, metric, SCORE_LEGEND_CORNER_OVERRIDES.get(task))
    _draw_noise_floor_panel(ax_noise, df, min_valid, metric)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance_appendix(
    df: pd.DataFrame, out_path: Path, min_valid: int = 3, sample_sizes: tuple[int, ...] = (3, 4, 5), metric: str = "score",
) -> None:
    """Paper appendix figure: completion and hurdle counts, and CI shrinkage."""
    fig, (ax_complete, ax_ci) = plt.subplots(1, 2, figsize=(11, 4.8))
    _draw_completion_panel(ax_complete, df)
    _draw_ci_panel(ax_ci, df, sample_sizes, metric)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(run_dir: Path, min_valid: int = 3, out_dir: Path | None = None, metric: str = "score") -> None:
    df = build_run_table([run_dir])
    task = df["task"].iloc[0]
    out_dir = (out_dir or REPO_ROOT / "analysis" / run_dir.name) / "variance"
    plot_variance_overview(df, out_dir / f"{task}_overview.pdf", min_valid, metric=metric)
    print(f"wrote {out_dir}/{task}_overview.pdf")


def plot_merged(run_dirs: list[Path], min_valid: int = 3, out_dir: Path | None = None, metric: str = "score") -> None:
    df = build_run_table(run_dirs)
    tasks = sorted(df["task"].unique())
    out_dir = (out_dir or REPO_ROOT / "analysis" / "+".join(d.name for d in run_dirs)) / "variance"
    plot_variance_faceted(df, out_dir / f"{'+'.join(tasks)}_overview.pdf", min_valid, metric=metric)
    print(f"wrote {out_dir}/{'+'.join(tasks)}_overview.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one or more grid run directories, e.g. runs/redshift-full-...")
    parser.add_argument(
        "--merge", action="store_true",
        help="one figure with one row per task (faceted, not pooled) instead of a separate figure set per grid dir.",
    )
    parser.add_argument(
        "--min-valid", type=int, default=3,
        help="min valid replicates/cell for the score-variance/noise-floor qualifying-cell cutoff (default 3)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="write into this directory's variance/ subfolder (set by analysis_single_task.py when chaining)",
    )
    parser.add_argument(
        "--metric", default="score", metavar="METRIC",
        help="column for the variance panels (default: score; gap_closed for multi-task data)",
    )
    args = parser.parse_args()
    if args.merge:
        plot_merged(args.run_dirs, args.min_valid, args.out_dir, args.metric)
    else:
        for run_dir in args.run_dirs:
            plot_grid(run_dir, args.min_valid, args.out_dir, args.metric)


if __name__ == "__main__":
    main()
