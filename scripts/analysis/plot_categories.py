"""Taxonomy figures: per-dimension category distributions and category share per axis level.

Writes analysis/taxonomy_<grid_label>/<dim>/*.pdf and report.txt. --summary adds summary/ with a
Cramer's V axis x dimension table (cramers_v.csv, association_table.tex) and
unconditional_distributions.pdf.

Usage: uv run python -m scripts.analysis.plot_categories --stage2-dir scripts/taxonomy/stage2_output/<combo> runs/<grid> [...]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import chi2_contingency

from axes import AXIS_LEVELS, AXIS_NAMES, display_level
from scripts.analysis.utils import build_taxonomy_table, grid_label, restrict_to_common_configs, tee_stdout_to_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Shared figure style (copied per script, not imported).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
})

def _muted(cmap: matplotlib.colors.Colormap, factor: float = 0.35) -> matplotlib.colors.Colormap:
    """cmap blended toward white by `factor`."""
    colors = cmap(np.linspace(0, 1, 256))
    colors[:, :3] = colors[:, :3] * (1 - factor) + factor
    return matplotlib.colors.ListedColormap(colors)


BAR_COLOR = "tab:blue"
# Valence colors (-1 red, 0 white, +1 blue): reversed vlag, muted.
VALENCE_CMAP = _muted(sns.color_palette("vlag_r", as_cmap=True))
OTHER_COLOR = "#b0b0b0"
# tab10 minus blue/gray (reserved above) -- fixed order, never cycled per-facet, so a category
# means the same color in every axis's cross-tab.
CATEGORY_PALETTE = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]
MAX_LABEL_WORDS = 3

# Display names for the paper (distinct from the prompt labels in stage1_extract.py).
DIMENSION_DISPLAY_NAME = {
    "result": "Result Classification",
    "error_category": "Failure Category",
    "execution_quality": "Execution Quality",
    "model_usage": "Model Usage",
    "planning_exploration": "Planning & Exploration Behaviour",
    "verification_behavior": "Verification Behaviour",
}

# Short names for tight spots (summary tick labels, table headers).
DIMENSION_ABBREV = {
    "result": "Result",
    "error_category": "Error Category",
    "execution_quality": "Execution Quality",
    "model_usage": "Model Usage",
    "planning_exploration": "Planning & Exploration",
    "verification_behavior": "Verification Behavior",
}

# Display names only; the axis key stays "harness".
AXIS_DISPLAY_NAME = {"harness": "Reasoning"}

# Display names for model levels.
MODEL_LEVEL_LABELS = {"qwen35-35b-a3b-fp8": "Small", "qwen35-122b-a10b-fp8": "Medium", "qwen35-397b-a17b-fp8": "Large",
                       "step37-198b-a11b-fp8": "Stepfun"}

# Family names for --hue-axis model.
MODEL_FAMILY_LABELS = {"qwen35-": "Qwen", "step37-": "StepFun"}


def _model_family_label(level: str) -> str:
    for prefix, name in MODEL_FAMILY_LABELS.items():
        if level.startswith(prefix):
            return name
    return _level_label("model", level)


def _axis_display(axis: str) -> str:
    return AXIS_DISPLAY_NAME.get(axis, axis.capitalize())


def _level_label(axis: str, level: str) -> str:
    if axis == "model":
        return MODEL_LEVEL_LABELS.get(level, display_level(axis, level).capitalize())
    if axis not in AXIS_LEVELS:
        return str(level).replace("-", " ").title()  # e.g. "task": "mmlu-astronomy" -> "Mmlu Astronomy"
    return display_level(axis, level).capitalize()


def _levels_present(axis: str, values) -> list:
    """Levels of `axis` present in `values`, in AXIS_LEVELS order, then unknown values sorted."""
    present = set(values)
    known = [lvl for lvl in AXIS_LEVELS[axis] if lvl in present] if axis in AXIS_LEVELS else []
    unknown = sorted(present - set(known))
    return known + unknown


def _split_two_lines(label: str) -> str:
    """Split a label into two lines of roughly equal word count."""
    words = label.split()
    if len(words) <= 1:
        return label
    mid = (len(words) + 1) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def _wrap(label: str) -> str:
    """Unwrapped for MAX_LABEL_WORDS words or fewer; past that, split into two lines (see
    _split_two_lines)."""
    if len(label.split()) <= MAX_LABEL_WORDS:
        return label
    return _split_two_lines(label)


def _wrap_bar_label(label: str) -> str:
    """One word per line if the second word is hyphenated, else _split_two_lines."""
    words = label.split()
    if len(words) <= 1:
        return label
    if len(words) >= 2 and "-" in words[1]:
        return "\n".join(words)
    return _split_two_lines(label)


def _fit_hue_labels(fig, ax, hue_display: list[str], n_levels: int) -> list[str]:
    """Set hue tick labels, wrapping or shrinking them until each fits its bar (measured after drawing)."""
    def fits(labels: list[str], fontsize: int) -> bool:
        ax.set_xticklabels(labels * n_levels, fontsize=fontsize)
        fig.canvas.draw()
        bar_widths_px = [p.get_window_extent().width for p in ax.patches[:len(labels) * n_levels]]
        label_widths_px = [t.get_window_extent().width for t in ax.get_xticklabels()]
        return all(lw <= bw for lw, bw in zip(label_widths_px, bar_widths_px))

    if fits(hue_display, 8):
        return hue_display
    wrapped = [_split_two_lines(h) for h in hue_display]
    fontsize = 9
    while fontsize > 6 and not fits(wrapped, fontsize):
        fontsize -= 1
    return wrapped


def plot_distribution(df: pd.DataFrame, stage2_dir: Path, dim: str, display_name: str, out_path: Path) -> None:
    """Bar chart of a dimension's categories, colored and sorted by valence if available.
    Counts come from df, not categories.json, which covers the full corpus.
    """
    categories = json.loads((stage2_dir / f"{dim}.categories.json").read_text())
    counts_in_df = df[dim].value_counts()
    for c in categories:
        c["count"] = int(counts_in_df.get(c["label"], 0))
    has_valence = bool(categories) and all("valence" in c for c in categories)
    categories = sorted(categories, key=(lambda c: c["valence"]) if has_valence else (lambda c: c["count"]))
    total = sum(c["count"] for c in categories)
    if total == 0:
        print(f"[{dim}] no runs in this subset have a category assigned -- skipping distribution plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(categories) + 1)))
    labels = [_wrap(c["label"]) for c in categories]
    counts = [c["count"] for c in categories]
    max_count = max(counts)
    colors = [VALENCE_CMAP((c["valence"] + 1) / 2) for c in categories] if has_valence else BAR_COLOR
    ax.barh(labels, counts, color=colors, height=0.6)
    for y, c in enumerate(categories):
        ax.text(c["count"] + max_count * 0.02, y, f"{c['count']} ({c['count'] / total:.0%})",
                 va="center", fontsize=10)
    ax.set_xlabel("# Occurrences")
    ax.set_title(f"Distribution of {display_name.lower()}")
    ax.set_xlim(0, max_count * 1.3)

    if has_valence:
        # Valence colorbar with ticks at -1, 0, +1.
        sm = plt.cm.ScalarMappable(cmap=VALENCE_CMAP, norm=plt.Normalize(-1, 1))
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_ticks([-1, 0, 1])
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Valence (LLM-judged)", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_share_by_axis(df: pd.DataFrame, dim: str, display_name: str, axis: str, out_path: Path,
                        top_n: int = 7, categories: list[dict] | None = None, hue_axis: str | None = None) -> None:
    """100%-stacked bars of category shares per level of `axis`; categories beyond top_n-1 fold into
    "Other". Valence sets color and order when available. hue_axis splits each level into one bar per
    hue level. Legend above the plot, worst to best.
    """
    sub = df[df[dim].notna()]
    overall = sub[dim].value_counts()
    top = list(overall.index[: top_n - 1])  # which categories stay distinct -- always by frequency
    has_valence = bool(categories) and all("valence" in c for c in categories)
    if has_valence:
        valence_of = {c["label"]: c["valence"] for c in categories}
        top = sorted(top, key=lambda label: valence_of.get(label, 0.0))  # worst-to-best -> bottom-to-top
    order = top + (["Other"] if len(overall) > len(top) else [])
    grouped = sub[dim].where(sub[dim].isin(top), "Other")

    levels = _levels_present(axis, sub[axis])

    if has_valence:
        color_of = {label: VALENCE_CMAP((valence_of.get(label, 0.0) + 1) / 2) for label in top} | {"Other": OTHER_COLOR}
    else:
        color_of = {label: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)] for i, label in enumerate(top)} | {"Other": OTHER_COLOR}

    if hue_axis is None:
        shares = pd.crosstab(sub[axis], grouped, normalize="index").reindex(index=levels, columns=order).fillna(0)
        fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(levels)) + 0.5, 5.0))
        bottom = np.zeros(len(levels))
        display_levels = [_level_label(axis, lvl) for lvl in levels]
        for label in order:
            vals = shares[label].to_numpy()
            # Hairline white edges separate segments.
            ax.bar(display_levels, vals, bottom=bottom, color=color_of[label], edgecolor="white", linewidth=0.8,
                   label=_wrap(label) if label != "Other" else "Other")
            text_color = "white" if matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color_of[label]))[2] < 0.6 else "black"
            for x, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 0.05:
                    ax.text(x, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=9, color=text_color)
            bottom += vals
        ax.set_xlabel(f"{_axis_display(axis)} Levels")
    else:
        hue_levels = _levels_present(hue_axis, sub[hue_axis])
        n_hue = len(hue_levels)
        idx = pd.MultiIndex.from_product([levels, hue_levels])
        shares = (pd.crosstab([sub[axis], sub[hue_axis]], grouped, normalize="index")
                  .reindex(index=idx, columns=order).fillna(0))

        group_width = 0.92  # white edges and the *0.9 factor below keep a seam between groups
        bar_width = group_width / n_hue
        x_base = np.arange(len(levels))
        x_positions = np.array([x_base[i] + (j - (n_hue - 1) / 2) * bar_width
                                 for i in range(len(levels)) for j in range(n_hue)])

        fig, ax = plt.subplots(figsize=(max(5, 1.3 * n_hue * len(levels)) + 0.5, 2.5))
        bottom = np.zeros(len(x_positions))
        for label in order:
            vals = shares[label].to_numpy()  # (level, hue) row order matches x_positions
            ax.bar(x_positions, vals, width=bar_width * 0.95, bottom=bottom, color=color_of[label],
                   edgecolor="white", linewidth=0.8, label=_wrap(label) if label != "Other" else "Other")
            text_color = "white" if matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color_of[label]))[2] < 0.6 else "black"
            for xp, v, b in zip(x_positions, vals, bottom):
                if v >= 0.05:
                    ax.text(xp, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=7, color=text_color,
                             transform=matplotlib.transforms.offset_copy(ax.transData, fig=fig, y=-1, units="points"))
            bottom += vals

        hue_display = [_model_family_label(h) if hue_axis == "model" else _level_label(hue_axis, h) for h in hue_levels]
        ax.set_xticks(x_positions)
        used_labels = _fit_hue_labels(fig, ax, hue_display, len(levels))
        # Offset below wrapped hue labels, derived from the rendered axes height so the gap is constant.
        wrapped_now = any("\n" in lbl for lbl in used_labels)
        fig.canvas.draw()
        axes_height_pts = ax.get_window_extent().height * 72.0 / fig.dpi
        group_gap_pts = 27 if wrapped_now else 17  # clearance below the hue subtick label(s)
        label_gap_pts = 5 if wrapped_now else -3  # clearance from group label to the x-axis title
        group_label_y = -group_gap_pts / axes_height_pts
        for i, lvl in enumerate(levels):
            ax.text(x_base[i], group_label_y, _level_label(axis, lvl), ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{_axis_display(axis)} Levels", labelpad=group_gap_pts + label_gap_pts)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of Occurrences")

    # Legend rows above the plot, worst to best; ncol halves until it fits the axes width.
    handles, labels = ax.get_legend_handles_labels()
    ncol = len(order)
    while True:
        legend = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
                            fontsize=9, columnspacing=1.2, handletextpad=0.5)
        fig.canvas.draw()
        legend_width = legend.get_window_extent().width
        axes_width = ax.get_window_extent().width
        if legend_width <= axes_width or ncol <= 1:
            break
        ncol = -(-ncol // 2)  # ceil division -- halve the row count, rounding up

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- Appendix summaries (--summary) ---------------------------------------------------------------

def _cramers_v(df: pd.DataFrame, axis: str, dim: str) -> float | None:
    """Cramer's V between `axis` and `dim`'s categories; None for a degenerate table."""
    sub = df[df[dim].notna()]
    table = pd.crosstab(sub[axis], sub[dim])
    if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() == 0:
        return None
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, c = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def association_table_data(df: pd.DataFrame, axes: list[str], dimensions: list[str]) -> pd.DataFrame:
    """One row per (axis, dimension): Cramer's V and its rank within the dimension."""
    v = np.full((len(axes), len(dimensions)), np.nan)
    for i, axis in enumerate(axes):
        for j, dim in enumerate(dimensions):
            val = _cramers_v(df, axis, dim)
            if val is not None:
                v[i, j] = val

    ranks = np.full_like(v, np.nan)
    for j in range(len(dimensions)):
        col = v[:, j]
        order = [i for i in np.argsort(-col) if not np.isnan(col[i])]
        for r, i in enumerate(order, start=1):
            ranks[i, j] = r

    rows = [
        {"axis": axis, "dimension": dim, "cramers_v": v[i, j],
         "rank": int(ranks[i, j]) if not np.isnan(ranks[i, j]) else None}
        for i, axis in enumerate(axes) for j, dim in enumerate(dimensions)
    ]
    return pd.DataFrame(rows)


def write_association_table_csv(data: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)


def write_association_table_tex(data: pd.DataFrame, out_path: Path) -> None:
    """Write the axis x dimension table (rank and V) as a booktabs LaTeX table."""
    axes = list(dict.fromkeys(data["axis"]))
    dimensions = list(dict.fromkeys(data["dimension"]))
    cell = {(row.axis, row.dimension): row for row in data.itertuples()}

    col_spec = "l" + "cc" * len(dimensions)
    header1 = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DIMENSION_ABBREV[d].replace('&', chr(92) + '&')}}}" for d in dimensions)
    cmidrules = " ".join(f"\\cmidrule(lr){{{2*j+2}-{2*j+3}}}" for j in range(len(dimensions)))
    header2 = "Axis & " + " & ".join("Rank & V" for _ in dimensions)

    rows = []
    for axis in axes:
        cells = []
        for dim in dimensions:
            row = cell[(axis, dim)]
            if row.rank is None or pd.isna(row.rank):
                cells.append("n/a & --")
            else:
                cells.append(f"{int(row.rank)} & {row.cramers_v:.2f}")
        rows.append(f"    {_axis_display(axis)} & " + " & ".join(cells) + r" \\")

    tex = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        "  \\caption{Association between each axis and each taxonomy dimension's category "
        "distribution (pooled across the three Qwen sizes, all four tasks). Axes are ranked within "
        "each dimension's own column (1 = strongest association) by Cram\\'{e}r's V, a "
        "chi-squared-based effect size for categorical variables (0 = no association, 1 = perfect "
        "association) -- the categorical-outcome counterpart to \\cref{tab:axis-effects}'s "
        "score-based ranking.}\n"
        "  \\label{tab:taxonomy-association}\n"
        "  \\small\n"
        "  \\resizebox{\\textwidth}{!}{%\n"
        f"  \\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
        "    \\toprule\n"
        f"    & {header1} \\\\\n"
        f"    {cmidrules}\n"
        f"    {header2} \\\\\n"
        "    \\midrule\n"
        + "\n".join(rows) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}%\n"
        "  }\n"
        "\\end{table}\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)


def plot_unconditional_distributions(df: pd.DataFrame, stage2_dir: Path, dimensions: list[str], out_path: Path) -> None:
    """One 100%-stacked bar per dimension with in-segment labels (share only below 8%, none below 4%).
    Counts come from df.
    """
    fig, ax = plt.subplots(figsize=(2.3 * len(dimensions) + 1, 7))
    x_positions = np.arange(len(dimensions))

    for x, dim in zip(x_positions, dimensions):
        categories = json.loads((stage2_dir / f"{dim}.categories.json").read_text())
        counts_in_df = df[df[dim].notna()][dim].value_counts()
        for c in categories:
            c["count"] = int(counts_in_df.get(c["label"], 0))
        has_valence = bool(categories) and all("valence" in c for c in categories)
        ordered = sorted(categories, key=(lambda c: c["valence"]) if has_valence else (lambda c: -c["count"]))
        total = sum(c["count"] for c in ordered)

        bottom = 0.0
        for k, c in enumerate(ordered):
            share = c["count"] / total if total else 0.0
            color = VALENCE_CMAP((c["valence"] + 1) / 2) if has_valence else CATEGORY_PALETTE[k % len(CATEGORY_PALETTE)]
            ax.bar(x, share, bottom=bottom, width=0.7, color=color, edgecolor="white", linewidth=0.8)
            if share >= 0.04:
                text_color = "white" if matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(color))[2] < 0.6 else "black"
                wrapped = _wrap_bar_label(c["label"])
                label = f"{wrapped}\n{share:.0%}" if share >= 0.08 else f"{share:.0%}"
                ax.text(x, bottom + share / 2, label, ha="center", va="center", fontsize=8.5, color=text_color)
            bottom += share

    ax.set_xticks(x_positions)
    ax.set_xticklabels([DIMENSION_ABBREV[d] for d in dimensions], rotation=0, ha="center")
    ax.set_xlim(-0.6, len(dimensions) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of Occurrences")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="runs/<grid> dirs matching stage2-dir's input grids")
    parser.add_argument("--dimensions", nargs="+", default=None, help="default: every dimension found in stage2-dir")
    parser.add_argument("--axes", nargs="+", default=list(AXIS_NAMES), choices=list(AXIS_NAMES) + ["task", "oracle"])
    parser.add_argument("--top-n", type=int, default=7, help="categories kept distinct in the axis cross-tab before folding into Other")
    parser.add_argument("--hue-axis", default=None, choices=list(AXIS_NAMES) + ["task", "oracle"],
                         help="split each level into one bar per level of this axis (also \"task\" or \"oracle\"); adds _x_<hue-axis> to filenames")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="override the default analysis/taxonomy_<grid_label>/ directory")
    parser.add_argument(
        "--common-configs-only", action="store_true",
        help="restrict to axis-level combinations every model has, so coverage differences (e.g. step37 lacks act-only) do not confound model comparisons",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="also write summary/: Cramer's V table (csv, tex) and unconditional distributions, over all axes and dimensions",
    )
    args = parser.parse_args()

    dimensions = args.dimensions or sorted(p.name.removesuffix(".categories.json") for p in args.stage2_dir.glob("*.categories.json"))
    df = build_taxonomy_table(args.run_dirs, args.stage2_dir, dimensions)
    # "oracle" from the grid directory name; keep in sync with analysis_cross_models.py.
    if "oracle" in args.axes or args.hue_axis == "oracle":
        df["oracle"] = np.where(df["grid"].str.contains("-oracle-"), "oracle", "baseline")
    # Computed before the tee, so report.txt captures every print.
    out_dir = args.out_dir or (REPO_ROOT / "analysis" / f"taxonomy_{grid_label(df)}")

    with tee_stdout_to_file(out_dir / "report.txt"):
        if args.common_configs_only:
            print("restricting to configs common to every model present (--common-configs-only)")
            df = restrict_to_common_configs(df)

        # One subdirectory per dimension.
        for dim in dimensions:
            display_name = DIMENSION_DISPLAY_NAME[dim]
            dim_dir = out_dir / dim
            plot_distribution(df, args.stage2_dir, dim, display_name, dim_dir / f"{dim}_categories.pdf")
            print(f"wrote {dim_dir / f'{dim}_categories.pdf'}")

        for dim in dimensions:
            display_name = DIMENSION_DISPLAY_NAME[dim]
            dim_dir = out_dir / dim
            categories = json.loads((args.stage2_dir / f"{dim}.categories.json").read_text())
            for axis in args.axes:
                if axis == args.hue_axis:
                    continue
                suffix = f"_x_{_axis_display(args.hue_axis).lower()}" if args.hue_axis else ""
                out_path = dim_dir / f"{dim}_by_{_axis_display(axis).lower()}{suffix}.pdf"
                plot_share_by_axis(df, dim, display_name, axis, out_path, args.top_n, categories, args.hue_axis)
                print(f"wrote {out_path}")

        if args.summary:
            summary_dir = out_dir / "summary"
            all_dims = sorted(p.name.removesuffix(".categories.json") for p in args.stage2_dir.glob("*.categories.json"))
            assoc_data = association_table_data(df, list(AXIS_NAMES), all_dims)
            csv_path = summary_dir / "cramers_v.csv"
            write_association_table_csv(assoc_data, csv_path)
            print(f"wrote {csv_path}")
            assoc_path = summary_dir / "association_table.tex"
            write_association_table_tex(assoc_data, assoc_path)
            print(f"wrote {assoc_path}")
            dist_path = summary_dir / "unconditional_distributions.pdf"
            plot_unconditional_distributions(df, args.stage2_dir, all_dims, dist_path)
            print(f"wrote {dist_path}")


if __name__ == "__main__":
    main()
