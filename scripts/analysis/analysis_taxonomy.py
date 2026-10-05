"""Join Stage 2 category assignments with the run table (utils.build_taxonomy_table), print
coverage and per-category counts, and write the table to CSV. --plots also runs plot_categories.py.

Usage:
  uv run python -m scripts.analysis.analysis_taxonomy runs/<grid> [<grid> ...] \
    --stage2-dir scripts/taxonomy/stage2_output/<combo> [--plots]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.utils import DropWrotePrefix, _section, build_taxonomy_table

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def print_coverage(df: pd.DataFrame, dimensions: list[str]) -> None:
    """How many of df's runs got a label per dimension -- a run can be missing one if Stage 1 found
    nothing scoreable for it, or if Stage 2 hasn't finished that dimension yet."""
    n = len(df)
    print(f"{n} runs total")
    for dim in dimensions:
        have = int(df[dim].notna().sum())
        print(f"  {dim:<24} {have:>5} / {n}  ({have / n:.1%})")


def print_category_stats(df: pd.DataFrame, stage2_dir: Path, dimensions: list[str]) -> None:
    """Per dimension: each category, its criterion and its count among df's runs."""
    for dim in dimensions:
        categories = json.loads((stage2_dir / f"{dim}.categories.json").read_text())
        counts = df[dim].value_counts()
        _section(f"{dim} categories")
        for cat in sorted(categories, key=lambda c: -counts.get(c["label"], 0)):
            n = int(counts.get(cat["label"], 0))
            valence = cat.get("valence")
            valence_str = f", valence {valence:+.1f}" if valence is not None else ""
            print(f"  {cat['label']} ({n}{valence_str})")
            print(f"      {cat['criterion']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="one or more grid run directories, e.g. runs/redshift-full-...")
    parser.add_argument("--stage2-dir", type=Path, required=True, help="scripts/taxonomy/stage2_output/<combo>")
    parser.add_argument("--out", type=Path, default=None, help="output CSV path (default: <stage2-dir>/combined.csv)")
    parser.add_argument("--plots", action="store_true",
                         help="also run plot_categories.py into analysis/taxonomy_<grid_label>/")
    args = parser.parse_args()

    dimensions = sorted(p.name.removesuffix(".categories.json") for p in args.stage2_dir.glob("*.categories.json"))
    df = build_taxonomy_table(args.run_dirs, args.stage2_dir, dimensions)
    print_coverage(df, dimensions)
    print_category_stats(df, args.stage2_dir, dimensions)

    out = args.out or (args.stage2_dir / "combined.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows, {len(df.columns)} columns)")

    if args.plots:
        _generate_plots(args.run_dirs, args.stage2_dir)


def _generate_plots(run_dirs: list[Path], stage2_dir: Path) -> None:
    """Run plot_categories.py's main() for these run dirs and stage2-dir."""
    import sys
    from scripts.analysis import plot_categories

    original_argv, original_stdout = sys.argv, sys.stdout
    try:
        sys.stdout = DropWrotePrefix(original_stdout)
        sys.argv = [plot_categories.__name__, "--stage2-dir", str(stage2_dir), *[str(d) for d in run_dirs]]
        plot_categories.main()
    finally:
        sys.argv = original_argv
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
